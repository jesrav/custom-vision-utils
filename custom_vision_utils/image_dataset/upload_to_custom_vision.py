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
from custom_vision_utils.sdk_helpers.helpers import get_tag_id


def _create_blob_image_file_entry(image: BlobImage, trainer, project_id):
    return ImageFileCreateEntry(
        name=image.name,
        contents=pil_image_to_byte_array(image.get_pil_image())
    )


def _create_blob_classier_image_file_entry(image: BlobClassifierImage, trainer, project_id):
    return ImageFileCreateEntry(
        contents=pil_image_to_byte_array(image.get_pil_image()),
        tag_ids=[
            get_tag_id(tag_name=tag_name, trainer=trainer, project_id=project_id)
            for tag_name in image.tag_names
        ]
    )


def _create_blob_object_detection_image_file_entry(image: BlobObjectDetectionImage, trainer, project_id):
    return ImageFileCreateEntry(
        contents=pil_image_to_byte_array(image.get_pil_image()),
        regions=[
            region.to_azure_region(tag_id=get_tag_id(region.tag_name, trainer, project_id))
            for region in image.regions
        ]
    )


def _create_local_image_file_entry(image: BlobImage, trainer, project_id):
    with open(image.uri, "rb") as image_contents:
        return ImageFileCreateEntry(
            name=image.name,
            contents=image_contents.read()
        )


def _create_local_classier_image_file_entry(image: BlobClassifierImage, trainer, project_id):
    with open(image.uri, "rb") as image_contents:
        return ImageFileCreateEntry(
            contents=image_contents.read(),
            tag_ids=[
                get_tag_id(tag_name=tag_name, trainer=trainer, project_id=project_id)
                for tag_name in image.tag_names
            ]
        )


def _create_local_object_detection_image_file_entry(image: BlobObjectDetectionImage, trainer, project_id):
    with open(image.uri, "rb") as image_contents:
        return ImageFileCreateEntry(
            contents=image_contents.read(),
            regions=[
                region.to_azure_region(tag_id=get_tag_id(region.tag_name, trainer, project_id))
                for region in image.regions
            ]
        )


def _create_image_file_entry(image, trainer, project_id):
    type_image_file_entry_map = {
        LocalImage: _create_local_image_file_entry,
        LocalClassifierImage: _create_local_classier_image_file_entry,
        LocalObjectDetectionImage: _create_local_object_detection_image_file_entry,
        BlobImage: _create_blob_image_file_entry,
        BlobClassifierImage: _create_blob_classier_image_file_entry,
        BlobObjectDetectionImage: _create_blob_object_detection_image_file_entry,
    }
    if type(image) not in type_image_file_entry_map:
        raise TypeError(
            f"image parameter must be an image object in one of the types: {type_image_file_entry_map.keys()}"
        )
    else:
        return type_image_file_entry_map[type(image)](image, trainer, project_id)


def _upload_image(image, trainer, project_id):
    """Upload a single image, including the name with metadata.

    It is faster to upload images in batches, but if we want to keep a name/id
    we need to add it as metadata and upload one image at a time."""
    _ = trainer.create_images_from_files(
        project_id,
        ImageFileCreateBatch(
            images=[_create_image_file_entry(image=image, trainer=trainer, project_id=project_id)],
            metadata={"name": image.name},
        )
    )


def upload_to_custom_vision(images, trainer, project_id):
    """Upload images to Custom vision project."""
    for image in track(images):
        _upload_image(image, trainer, project_id)

