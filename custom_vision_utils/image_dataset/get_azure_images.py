import math
from typing import List, Iterable

from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry

from custom_vision_utils.image import (
    LocalImage,
    LocalClassifierImage,
    LocalObjectDetectionImage,
    BlobImage,
    BlobClassifierImage,
    BlobObjectDetectionImage,
)
from custom_vision_utils.image_dataset.image_dataset_interface import ImageDataSetInterface
from custom_vision_utils.pillow_utils import pil_image_to_byte_array


def _create_image_file_entry(image):
    if type(image) in [LocalImage, LocalClassifierImage, LocalObjectDetectionImage]:
        with open(image.uri, "rb") as image_contents:
            return ImageFileCreateEntry(
                name=image.uri.stem,
                contents=image_contents.read()
            )
    elif type(image) in [BlobImage, BlobClassifierImage, BlobObjectDetectionImage]:
        return ImageFileCreateEntry(
            name=image.name,
            contents=pil_image_to_byte_array(image.get_pil_image())
        )
    else:
        raise TypeError(
            "image must be an image object in one of the types:"
            "[LocalImage, LocalClassifierImage, LocalObjectDetectionImage, "
            "BlobImage, BlobClassifierImage, BlobObjectDetectionImage]"
        )


def _get_azure_images(
        image_data_set: ImageDataSetInterface,
        start_inx: int,
        stop_inx: int,
) -> List[ImageFileCreateEntry]:
    """Get images in Azure specific format.

    :param image_data_set: Image data set
    :param start_inx: Start index of images to generate from self.images
    :param stop_inx: Stop index of images to generate from self.images
    :return:
    """
    if not start_inx:
        start_inx = 0
    if not stop_inx:
        stop_inx = len(image_data_set)

    return [_create_image_file_entry(image) for image in image_data_set[start_inx:stop_inx]]


def get_azure_image_batches(
        image_data_set: ImageDataSetInterface,
        batch_size: int = 64,
) -> Iterable[List[ImageFileCreateEntry]]:
    """
    creates a generator that yields batches of images in Azure specific format of size batch_size. Custom Vision
    can maximally upload batches of 64.
    """
    num_batches = math.ceil(len(image_data_set) / batch_size)
    for i in range(num_batches):
        start_idx = i * batch_size
        stop_inx = start_idx + batch_size
        yield _get_azure_images(
            image_data_set=image_data_set,
            start_inx=start_idx,
            stop_inx=stop_inx,
        )
