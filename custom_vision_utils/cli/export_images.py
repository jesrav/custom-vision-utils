from pathlib import Path
from typing import Union, Type

import click
from rich.progress import track
from dotenv import load_dotenv, find_dotenv

from custom_vision_utils.cli.logger import logger
from custom_vision_utils.sdk_helpers import get_trainer, get_project_id
from custom_vision_utils.image import (
    LocalImage,
    LocalClassifierImage,
    LocalObjectDetectionImage,
    BlobImage,
    BlobClassifierImage,
    BlobObjectDetectionImage,
)
from custom_vision_utils.image_dataset import (
    LocalImageDataSet,
    LocalClassifierDataSet,
    LocalObjectDetectionDataSet,
    BlobImageDataSet,
    BlobClassifierDataSet,
    BlobObjectDetectionDataSet,
)

load_dotenv((find_dotenv()))

SUPPORTED_DOMAIN_TYPES = ["Classification", "ObjectDetection", None]

ImageDataSet = Union[
    LocalImageDataSet,
    BlobImageDataSet,
    LocalClassifierDataSet,
    BlobClassifierDataSet,
    LocalObjectDetectionDataSet,
    BlobObjectDetectionDataSet,
]

ImageClass = Union[
    Type[BlobImage],
    Type[LocalImage],
    Type[BlobClassifierImage],
    Type[LocalClassifierImage],
    Type[BlobObjectDetectionImage],
    Type[LocalObjectDetectionImage],
]


class DomainTypeNotSupportedError(BaseException):
    pass


def get_domain_type_of_project(trainer, project_name: str) -> str:
    project = trainer.get_project(get_project_id(trainer, project_name))
    project_domain_id = project.settings.domain_id
    return trainer.get_domain(project_domain_id).type


def select_image_data_set(
    domain_type: str, container_name_chosen: bool
) -> ImageDataSet:
    if container_name_chosen and domain_type is None:
        return BlobImageDataSet()
    elif domain_type is None:
        return LocalImageDataSet()
    elif domain_type == "Classification" and container_name_chosen:
        return BlobClassifierDataSet()
    elif domain_type == "Classification":
        return LocalClassifierDataSet()
    elif domain_type == "ObjectDetection" and container_name_chosen:
        return BlobObjectDetectionDataSet()
    else:
        return LocalObjectDetectionDataSet()


def select_image_class(domain_type: str, container_name_chosen: bool) -> ImageClass:
    if container_name_chosen and domain_type is None:
        return BlobImage
    elif domain_type is None:
        return LocalImage
    elif domain_type == "Classification" and container_name_chosen:
        return BlobClassifierImage
    elif domain_type == "Classification":
        return LocalClassifierImage
    elif domain_type == "ObjectDetection" and container_name_chosen:
        return BlobObjectDetectionImage
    else:
        return LocalObjectDetectionImage


def get_image_data_from_custom_vision(
    project_name: str,
    image_outdir: str,
    container_name: str,
    untagged: bool,
) -> ImageDataSet:
    batch_size = 50
    trainer = get_trainer()
    project_id = get_project_id(trainer, project_name)

    domain_type = get_domain_type_of_project(trainer, project_name)
    if untagged:
        domain_type = None
    if domain_type not in SUPPORTED_DOMAIN_TYPES:
        raise DomainTypeNotSupportedError(
            f"Domain type {domain_type} not supported. Only {SUPPORTED_DOMAIN_TYPES} can be used."
        )
    container_name_chosen = container_name is not None

    image_data = select_image_data_set(domain_type, container_name_chosen)
    image_class = select_image_class(domain_type, container_name_chosen)

    image_list_length = 1
    iteration_idx = 0
    while image_list_length != 0:
        custom_vision_images = trainer.get_images(
            project_id=project_id,
            take=batch_size,
            skip=iteration_idx * batch_size,
            tagging_status="Tagged" if not untagged else "Untagged",
        )
        if len(custom_vision_images) == 0:
            if iteration_idx == 0:
                logger.info("No images to export.")
            break
        logger.info(
            f"Exporting batch number {iteration_idx + 1} of object detection "
            f"images with tagged object regions to {image_outdir}."
        )
        for cv_image in track(custom_vision_images):
            image = image_class.from_azure_custom_vision_image(
                custom_vision_image=cv_image,
                folder=image_outdir,
                container_name=container_name,
            )
            image_data.append(image)

        iteration_idx += 1
    return image_data


@click.command()
@click.argument("project-name", type=click.Path())
@click.argument("image-outdir", type=click.Path(exists=True, file_okay=False))
@click.option("--container-name", type=click.Path())
@click.option("--data-config-outpath", type=click.Path())
@click.option(
    "--untagged",
    type=click.BOOL,
    is_flag=True,
    default=False,
)
def export_images(
    project_name, image_outdir, container_name, data_config_outpath, untagged
) -> None:
    image_data = get_image_data_from_custom_vision(
        project_name=project_name,
        image_outdir=image_outdir,
        container_name=container_name,
        untagged=untagged,
    )
    if data_config_outpath and image_data:
        image_data.write_config(outfile=Path(data_config_outpath))
