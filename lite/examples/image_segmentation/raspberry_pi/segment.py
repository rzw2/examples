# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run image segmentation."""

import argparse
import sys
import time
from typing import List

import cv2
from image_segmenter import ColoredLabel
from image_segmenter import ImageSegmenter
from image_segmenter import ImageSegmenterOptions
import numpy as np
import utils

import ctypes
from ctypes.util import find_library
tmp_lib = find_library("gomp")
if tmp_lib != None:
    # This is a hacky way to get on camera python working, as OpenMP linking isn't working
    ctypes.CDLL(tmp_lib, mode=ctypes.RTLD_GLOBAL)
    import chronoptics.tof as tof

# Visualization parameters
_FPS_AVERAGE_FRAME_COUNT = 10
_FPS_LEFT_MARGIN = 24  # pixels
_LEGEND_TEXT_COLOR = (0, 0, 255)  # red
_LEGEND_BACKGROUND_COLOR = (255, 255, 255)  # white
_LEGEND_FONT_SIZE = 1
_LEGEND_FONT_THICKNESS = 1
_LEGEND_ROW_SIZE = 20  # pixels
_LEGEND_RECT_SIZE = 16  # pixels
_LABEL_MARGIN = 10
_OVERLAY_ALPHA = 0.5
_PADDING_WIDTH_FOR_LEGEND = 150  # pixels


def run(model: str, display_mode: str, num_threads: int, enable_edgetpu: bool,
        camera_id: int, width: int, height: int) -> None:
    """Continuously run inference on images acquired from the camera.

    Args:
        model: Name of the TFLite image segmentation model.
        display_mode: Name of mode to display image segmentation.
        num_threads: Number of CPU threads to run the model.
        enable_edgetpu: Whether to run the model on EdgeTPU.
        camera_id: The camera id to be passed to OpenCV.
        width: The width of the frame captured from the camera.
        height: The height of the frame captured from the camera.
    """

    # Initialize the image segmentation model.
    options = ImageSegmenterOptions(
        num_threads=num_threads, enable_edgetpu=enable_edgetpu)
    segmenter = ImageSegmenter(model_path=model, options=options)

    # Variables to calculate FPS
    counter, fps = 0, 0
    start_time = time.time()

    # Need to add these as inputs to the script
    dmax = 15.0
    fps = 8
    channel = 0
    qvga = False

    # Use the Chronoptics Kea camera instead of the /dev/video1
    proc = tof.ProcessingConfig()

    user = tof.UserConfig()
    # Example of using the camera configuration
    user.setFps(fps)
    user.setMaxDistance(dmax)
    user.setEnvironment(tof.ImagingEnvironment.INSIDE)
    user.setIntegrationTime(tof.IntegrationTime.MEDIUM)
    user.setStrategy(tof.Strategy.BALANCED)
    user.setChannel(channel)

    cam = tof.EmbeddedKeaCamera(proc)
    #cam = tof.KeaCamera(proc, "")
    config = user.toCameraConfig(cam)

    if qvga:
        for n in range(0, config.frameSize()):
            config.setBinning(n, 1)
    proc_update = config.defaultProcessing()
    cam.setCameraConfig(config)
    cam.setProcessConfig(proc_update)

    # Start capturing video input from the camera
    #cap = cv2.VideoCapture(camera_id)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    tof.selectStreams(cam, [tof.FrameType.BGR_PROJECTED])
    cam.start()

    # Continuously capture images from the camera and run inference.
    while cam.isStreaming():
        frames = cam.getFrames()
        image = np.asarray(frames[0])

        counter += 1
        image = cv2.flip(image, 0)

        # Segment with each frame from camera.
        segmentation_result = segmenter.segment(image)

        # Convert the segmentation result into an image.
        seg_map_img, found_colored_labels = utils.segmentation_map_to_image(
            segmentation_result)

        # Resize the segmentation mask to be the same shape as input image.
        seg_map_img = cv2.resize(
            seg_map_img,
            dsize=(image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST)

        # Visualize segmentation result on image.
        overlay = visualize(image, seg_map_img, display_mode, fps,
                            found_colored_labels)

        # Calculate the FPS
        if counter % _FPS_AVERAGE_FRAME_COUNT == 0:
            end_time = time.time()
            fps = _FPS_AVERAGE_FRAME_COUNT / (end_time - start_time)
            start_time = time.time()

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('image_segmentation', overlay)

    cam.stop()
    cv2.destroyAllWindows()


def visualize(input_image: np.ndarray, segmentation_map_image: np.ndarray,
              display_mode: str, fps: float,
              colored_labels: List[ColoredLabel]) -> np.ndarray:
    """Visualize segmentation result on image.

    Args:
        input_image: The [height, width, 3] RGB input image.
        segmentation_map_image: The [height, width, 3] RGB segmentation map image.
        display_mode: How the segmentation map should be shown. 'overlay' or
          'side-by-side'.
        fps: Value of fps.
        colored_labels: List of colored labels found in the segmentation result.

    Returns:
        Input image overlaid with segmentation result.
    """
    # Show the input image and the segmentation map image.
    if display_mode == 'overlay':
        # Overlay mode.
        overlay = cv2.addWeighted(input_image, _OVERLAY_ALPHA,
                                  segmentation_map_image, _OVERLAY_ALPHA, 0)
    elif display_mode == 'side-by-side':
        # Side by side mode.
        overlay = cv2.hconcat([input_image, segmentation_map_image])
    else:
        sys.exit(f'ERROR: Unsupported display mode: {display_mode}.')

    # Show the FPS
    fps_text = 'FPS = ' + str(int(fps))
    text_location = (_FPS_LEFT_MARGIN, _LEGEND_ROW_SIZE)
    cv2.putText(overlay, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                _LEGEND_FONT_SIZE, _LEGEND_TEXT_COLOR, _LEGEND_FONT_THICKNESS)

    # Initialize the origin coordinates of the label.
    legend_x = overlay.shape[1] + _LABEL_MARGIN
    legend_y = overlay.shape[0] // _LEGEND_ROW_SIZE + _LABEL_MARGIN

    # Expand the frame to show the label.
    overlay = cv2.copyMakeBorder(overlay, 0, 0, 0, _PADDING_WIDTH_FOR_LEGEND,
                                 cv2.BORDER_CONSTANT, None,
                                 _LEGEND_BACKGROUND_COLOR)

    # Show the label on right-side frame.
    for colored_label in colored_labels:
        rect_color = colored_label.color
        start_point = (legend_x, legend_y)
        end_point = (legend_x + _LEGEND_RECT_SIZE,
                     legend_y + _LEGEND_RECT_SIZE)
        cv2.rectangle(overlay, start_point, end_point, rect_color,
                      -_LEGEND_FONT_THICKNESS)

        label_location = legend_x + _LEGEND_RECT_SIZE + \
            _LABEL_MARGIN, legend_y + _LABEL_MARGIN
        cv2.putText(overlay, colored_label.label, label_location,
                    cv2.FONT_HERSHEY_PLAIN, _LEGEND_FONT_SIZE, _LEGEND_TEXT_COLOR,
                    _LEGEND_FONT_THICKNESS)
        legend_y += (_LEGEND_RECT_SIZE + _LABEL_MARGIN)

    return overlay


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Name of image segmentation model.',
        required=False,
        default='deeplabv3.tflite')
    parser.add_argument(
        '--displayMode',
        help='Mode to display image segmentation.',
        required=False,
        default='overlay')
    parser.add_argument(
        '--numThreads',
        help='Number of CPU threads to run the model.',
        required=False,
        default=4)
    parser.add_argument(
        '--enableEdgeTPU',
        help='Whether to run the model on EdgeTPU.',
        action='store_true',
        required=False,
        default=False)
    parser.add_argument(
        '--cameraId', help='Id of camera.', required=False, default=0)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        default=640)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        default=480)
    args = parser.parse_args()

    run(args.model, args.displayMode, int(args.numThreads),
        bool(args.enableEdgeTPU), int(args.cameraId), args.frameWidth,
        args.frameHeight)


if __name__ == '__main__':
    main()
