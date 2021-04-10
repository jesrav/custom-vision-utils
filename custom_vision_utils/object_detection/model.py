"""Wrapper class for a Custom Vision Classification model exported as a tensorflow model.

The code is basically copy pasted from examples in the Custom Vision documentation, with minor tweaks.
"""
import math
from pathlib import Path
from typing import Union, List

import numpy as np
import tensorflow as tf
from PIL import Image

from custom_vision_utils.object_detection.result import (
    ObjectDetectionResult,
    BoundingBox,
)


class ObjectDetectionModel(object):
    """Wrapper class for a Custom Vision object detection model exported as a tensorflow model"""

    ANCHORS = np.array(
        [[0.573, 0.677], [1.87, 2.06], [3.34, 5.47], [7.88, 3.53], [9.77, 9.17]]
    )
    IOU_THRESHOLD = 0.45
    DEFAULT_INPUT_SIZE = 512 * 512

    def __init__(
        self,
        model_filepath: Union[str, Path],
        labels_filepath: Union[str, Path],
        prob_threshold: float = 0.10,
        max_detections: int = 20,
    ):
        """Initialize model object

        :param model_filepath: Path to the exported model (extension .pb)
        :param labels_filepath: Path to the exported model labels (extension .txt)
        :param prob_threshold: threshold for class probability.
        :param max_detections: the max number of output custom_vision_results.
        """
        # Load graph
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v2.io.gfile.GFile(model_filepath, "rb") as f:
            graph_def.ParseFromString(f.read())

        # Load labels
        with open(labels_filepath, "r") as f:
            labels = [l.strip() for l in f.readlines()]

        assert len(labels) >= 1, "At least 1 label is required"

        self.labels = labels
        self.prob_threshold = prob_threshold
        self.max_detections = max_detections
        self.graph = tf.compat.v1.Graph()
        with self.graph.as_default():
            input_data = tf.compat.v1.placeholder(
                tf.float32, [1, None, None, 3], name="Placeholder"
            )
            tf.import_graph_def(
                graph_def, input_map={"Placeholder:0": input_data}, name=""
            )

    def _logistic(self, x):
        return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def _non_maximum_suppression(self, boxes, class_probs, max_detections):
        """Remove overlapping bouding boxes"""
        assert len(boxes) == len(class_probs)

        max_detections = min(max_detections, len(boxes))
        max_probs = np.amax(class_probs, axis=1)
        max_classes = np.argmax(class_probs, axis=1)

        areas = boxes[:, 2] * boxes[:, 3]

        selected_boxes = []
        selected_classes = []
        selected_probs = []

        while len(selected_boxes) < max_detections:
            # Select the prediction with the highest probability.
            i = np.argmax(max_probs)
            if max_probs[i] < self.prob_threshold:
                break

            # Save the selected prediction
            selected_boxes.append(boxes[i])
            selected_classes.append(max_classes[i])
            selected_probs.append(max_probs[i])

            box = boxes[i]
            other_indices = np.concatenate((np.arange(i), np.arange(i + 1, len(boxes))))
            other_boxes = boxes[other_indices]

            # Get overlap between the 'box' and 'other_boxes'
            x1 = np.maximum(box[0], other_boxes[:, 0])
            y1 = np.maximum(box[1], other_boxes[:, 1])
            x2 = np.minimum(box[0] + box[2], other_boxes[:, 0] + other_boxes[:, 2])
            y2 = np.minimum(box[1] + box[3], other_boxes[:, 1] + other_boxes[:, 3])
            w = np.maximum(0, x2 - x1)
            h = np.maximum(0, y2 - y1)

            # Calculate Intersection Over Union (IOU)
            overlap_area = w * h
            iou = overlap_area / (areas[i] + areas[other_indices] - overlap_area)

            # Find the overlapping predictions
            overlapping_indices = other_indices[np.where(iou > self.IOU_THRESHOLD)[0]]
            overlapping_indices = np.append(overlapping_indices, i)

            # Set the probability of overlapping predictions to zero, and udpate max_probs and max_classes.
            class_probs[overlapping_indices, max_classes[i]] = 0
            max_probs[overlapping_indices] = np.amax(
                class_probs[overlapping_indices], axis=1
            )
            max_classes[overlapping_indices] = np.argmax(
                class_probs[overlapping_indices], axis=1
            )

        assert len(selected_boxes) == len(selected_classes) and len(
            selected_boxes
        ) == len(selected_probs)
        return selected_boxes, selected_classes, selected_probs

    def _extract_bb(self, prediction_output, anchors):
        assert len(prediction_output.shape) == 3
        num_anchor = anchors.shape[0]
        height, width, channels = prediction_output.shape
        assert channels % num_anchor == 0

        num_class = int(channels / num_anchor) - 5
        assert num_class == len(self.labels)

        outputs = prediction_output.reshape((height, width, num_anchor, -1))

        # Extract bouding box information
        x = (
            self._logistic(outputs[..., 0])
            + np.arange(width)[np.newaxis, :, np.newaxis]
        ) / width
        y = (
            self._logistic(outputs[..., 1])
            + np.arange(height)[:, np.newaxis, np.newaxis]
        ) / height
        w = np.exp(outputs[..., 2]) * anchors[:, 0][np.newaxis, np.newaxis, :] / width
        h = np.exp(outputs[..., 3]) * anchors[:, 1][np.newaxis, np.newaxis, :] / height

        # (x,y) in the network outputs is the center of the bounding box. Convert them to top-left.
        x = x - w / 2
        y = y - h / 2
        boxes = np.stack((x, y, w, h), axis=-1).reshape(-1, 4)

        # Get confidence for the bounding boxes.
        objectness = self._logistic(outputs[..., 4])

        # Get class probabilities for the bounding boxes.
        class_probs = outputs[..., 5:]
        class_probs = np.exp(
            class_probs - np.amax(class_probs, axis=3)[..., np.newaxis]
        )
        class_probs = (
            class_probs
            / np.sum(class_probs, axis=3)[..., np.newaxis]
            * objectness[..., np.newaxis]
        )
        class_probs = class_probs.reshape(-1, num_class)

        assert len(boxes) == len(class_probs)
        return (boxes, class_probs)

    def _update_orientation(self, image):
        """
        corrects pillow_utils orientation according to EXIF data
        pillow_utils: input PIL pillow_utils
        returns corrected PIL pillow_utils
        """
        if hasattr(image, "_getexif"):
            exif = image._getexif()
            exif_orientation_tag = 0x0112
            if exif != None and exif_orientation_tag in exif:
                orientation = exif.get(exif_orientation_tag, 1)
                print("Image has EXIF Orientation: {}".format(str(orientation)))
                # orientation is 1 based, shift to zero based and flip/transpose based on 0-based values
                orientation -= 1
                if orientation >= 4:
                    image = image.transpose(Image.TRANSPOSE)
                if orientation in [2, 3, 6, 7]:
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                if orientation in [1, 2, 5, 6]:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

    def _preprocess(self, image):
        image = image.convert("RGB") if image.mode != "RGB" else image
        image = self._update_orientation(image)

        ratio = math.sqrt(self.DEFAULT_INPUT_SIZE / image.width / image.height)
        new_width = int(image.width * ratio)
        new_height = int(image.height * ratio)
        new_width = 32 * round(new_width / 32)
        new_height = 32 * round(new_height / 32)
        image = image.resize((new_width, new_height))
        return image

    def _predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float)[
            :, :, (2, 1, 0)
        ]  # RGB -> BGR

        with tf.compat.v1.Session(graph=self.graph) as sess:
            output_tensor = sess.graph.get_tensor_by_name("model_outputs:0")
            outputs = sess.run(
                output_tensor, {"Placeholder:0": inputs[np.newaxis, ...]}
            )
            return outputs[0]

    def _postprocess(self, prediction_outputs):
        """Extract bounding boxes from the model outputs.

        Args:
            prediction_outputs: Output from the object detection model. (H x W x C)

        Returns:
            List of Prediction objects.
        """
        boxes, class_probs = self._extract_bb(prediction_outputs, self.ANCHORS)

        # Remove bounding boxes whose confidence is lower than the threshold.
        max_probs = np.amax(class_probs, axis=1)
        (index,) = np.where(max_probs > self.prob_threshold)
        index = index[(-max_probs[index]).argsort()]

        # Remove overlapping bounding boxes
        (
            selected_boxes,
            selected_classes,
            selected_probs,
        ) = self._non_maximum_suppression(
            boxes[index], class_probs[index], self.max_detections
        )

        return [
            ObjectDetectionResult(
                tag_name=self.labels[selected_classes[i]],
                probability=round(float(selected_probs[i]), 8),
                bounding_box=BoundingBox(
                    left=round(float(selected_boxes[i][0]), 8),
                    top=round(float(selected_boxes[i][1]), 8),
                    width=round(float(selected_boxes[i][2]), 8),
                    height=round(float(selected_boxes[i][3]), 8),
                ),
            )
            for i in range(len(selected_boxes))
        ]

    def predict_image(self, image: Image) -> List[ObjectDetectionResult]:
        """Do prediction for an image.

        :param image: Pillow image
        :return: List of object detection predictions
        """
        inputs = self._preprocess(image)
        prediction_outputs = self._predict(inputs)
        return self._postprocess(prediction_outputs)