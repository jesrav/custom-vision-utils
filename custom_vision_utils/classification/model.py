"""Wrapper class for a Custom Vision Classification model exported as a tensorflow model.

The code is basically copy pasted from examples in the Custom Vision documentation, with minor tweaks.
"""
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from custom_vision_utils.classification.result import ImageClassifierResult


class ImageClassifierModel:
    """Wrapper class for a Custom Vision Classification model exported as a tensorflow model"""
    INPUT_TENSOR_NAME = "Placeholder:0"
    OUTPUT_LAYER = "loss:0"
    INPUT_NODE = "Placeholder:0"

    def __init__(self, model_filepath: Union[str, Path], labels_filepath: Union[str, Path]):
        """Initialize model object

        :param model_filepath: Path to the exported model (extension .pb)
        :param labels_filepath: Path to the exported model labels (extension .txt)
        """
        graph_def = tf.compat.v1.GraphDef()
        with open(model_filepath, "rb") as f:
            graph_def.ParseFromString(f.read())

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name="")

        # Get input shape
        with tf.compat.v1.Session(graph=self.graph) as sess:
            self.input_shape = sess.graph.get_tensor_by_name(
                self.INPUT_TENSOR_NAME
            ).shape.as_list()[1]

        with open(labels_filepath) as f:
            self.labels = [l.strip() for l in f.readlines()]

    @staticmethod
    def _convert_to_opencv(image):
        # RGB -> BGR conversion is performed as well.
        image = image.convert("RGB")
        r, g, b = np.array(image).T
        return np.array([b, g, r]).transpose()

    @staticmethod
    def _crop_center(img, cropx, cropy):
        h, w = img.shape[:2]
        startx = w // 2 - (cropx // 2)
        starty = h // 2 - (cropy // 2)
        return img[starty : starty + cropy, startx : startx + cropx]

    @staticmethod
    def _resize_down_to_1600_max_dim(image):
        h, w = image.shape[:2]
        if h < 1600 and w < 1600:
            return image

        new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
        return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _resize_to_256_square(image):
        h, w = image.shape[:2]
        return cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _update_orientation(image):
        if hasattr(image, "_getexif"):
            exif = image._getexif()
            exif_orientation_tag = 0x0112
            if exif != None and exif_orientation_tag in exif:
                orientation = exif.get(exif_orientation_tag, 1)
                # orientation is 1 based, shift to zero based and flip/transpose based on 0-based values
                orientation -= 1
                if orientation >= 4:
                    image = image.transpose(Image.TRANSPOSE)
                if orientation in [2, 3, 6, 7]:
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                if orientation in [1, 2, 5, 6]:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

    def predict_image(self, image: Image) -> List[ImageClassifierResult]:
        """Do prediction for an image.

        :param image: Pillow image
        :return: List of predictions
        """
        image = self._convert_to_opencv(image)
        image = self._resize_down_to_1600_max_dim(image)
        h, w = image.shape[:2]
        min_dim = min(w, h)
        max_square_image = self._crop_center(image, min_dim, min_dim)
        augmented_image = self._resize_to_256_square(max_square_image)
        augmented_image = self._crop_center(
            augmented_image, self.input_shape, self.input_shape
        )

        with tf.compat.v1.Session(graph=self.graph) as sess:
            prob_tensor = sess.graph.get_tensor_by_name(self.OUTPUT_LAYER)
            predictions = sess.run(
                prob_tensor, {self.INPUT_TENSOR_NAME: [augmented_image]}
            )[0]

        return [
            ImageClassifierResult(probability=float(proba), tag_name=label)
            for proba, label in zip(predictions, self.labels)
        ]
