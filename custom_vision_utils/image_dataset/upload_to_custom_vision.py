from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry, ImageFileCreateBatch
from rich.progress import track

from custom_vision_utils.image import (
    LocalImage,
    LocalClassifierImage,
    LocalObjectDetectionImage,
    BlobImage,
    BlobClassifierImage,
    BlobObjectDetectionImage,
)
from custom_vision_utils.pillow_utils import pil_image_to_byte_array


def _create_image_file_entry(image):
    if type(image) in [LocalImage, LocalClassifierImage, LocalObjectDetectionImage]:
        with open(image.uri, "rb") as image_contents:
            return ImageFileCreateEntry(
                name=image.name,
                contents=image_contents.read(),
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


def _upload_image(image, trainer, project_id):
    """Upload a single image, including the name with metadata.

    It is faster to upload images in batches, but if we want to keep a name/id
    we need to add it as metadata and upload one image at a time."""
    _ = trainer.create_images_from_files(
        project_id,
        ImageFileCreateBatch(
            images=[_create_image_file_entry(image=image)],
            metadata={"name": image.name},
        )
    )


def upload_to_custom_vision(images, trainer, project_id):
    """Upload images to Custom vision project."""
    for image in track(images):
        _upload_image(image, trainer, project_id)

