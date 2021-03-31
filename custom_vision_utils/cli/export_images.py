from pathlib import Path

import click
from rich.progress import track
from dotenv import load_dotenv, find_dotenv

from custom_vision_utils.cli.logger import logger
from custom_vision_utils.sdk_helpers import get_trainer, get_project_id
from custom_vision_utils.image import LocalImage, LocalClassifierImage, LocalObjectDetectionImage
from custom_vision_utils.image_dataset import LocalImageDataSet, LocalClassifierDataSet, LocalObjectDetectionDataSet


load_dotenv((find_dotenv()))


class DomainTypeNotSupportedError(BaseException):
    pass


def get_domain_type_of_project(trainer, project_name):
    project = trainer.get_project(get_project_id(trainer, project_name))
    project_domain_id = project.settings.domain_id
    return trainer.get_domain(project_domain_id).type


@click.command()
@click.argument("project-name", type=click.Path())
@click.argument("image-outdir", type=click.Path(exists=True, file_okay=False))
@click.option("--data-config-outpath", type=click.Path())
@click.option(
    "--untagged",
    type=click.BOOL,
    is_flag=True,
    default=False,
)
def export_images(
    project_name, image_outdir, data_config_outpath, untagged
):
    batch_size = 50
    trainer = get_trainer()
    project_id = get_project_id(trainer, project_name)

    domain_type = get_domain_type_of_project(trainer, project_name)
    if untagged:
        image_data = LocalImageDataSet()
        image_class = LocalImage
    else:
        if domain_type == 'Classification':
            image_data = LocalClassifierDataSet()
            image_class = LocalClassifierImage
        elif domain_type == 'ObjectDetection':
            image_data = LocalObjectDetectionDataSet()
            image_class = LocalObjectDetectionImage
        else:
            raise DomainTypeNotSupportedError(
                f"Domain type {domain_type} not supported. Only 'Classification' and 'ObjectDetection'."
            )

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
        for image in track(custom_vision_images):
            classifier_image = image_class.from_azure_custom_vision_image(
                custom_vision_image=image,
                folder=image_outdir,
            )
            image_data.append(classifier_image)

        image_list_length = len(custom_vision_images)
        iteration_idx += 1

    if data_config_outpath and image_data:
        image_data.write_config(outfile=Path(data_config_outpath))

