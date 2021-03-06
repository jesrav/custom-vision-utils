import click

from dotenv import load_dotenv
from pydantic import ValidationError

from custom_vision_utils.image_dataset.upload_to_custom_vision import (
    upload_to_custom_vision,
)
from custom_vision_utils.sdk_helpers import get_trainer, get_project_id
from custom_vision_utils.image_dataset import (
    LocalImageDataSet,
    LocalClassifierDataSet,
    LocalObjectDetectionDataSet,
    BlobImageDataSet,
    BlobClassifierDataSet,
    BlobObjectDetectionDataSet,
)
from custom_vision_utils.cli.logger import logger

image_data_classes = [
    LocalClassifierDataSet,
    LocalObjectDetectionDataSet,
    LocalImageDataSet,
    BlobClassifierDataSet,
    BlobObjectDetectionDataSet,
    BlobImageDataSet,
]


class BadDataConfigurationError(BaseException):
    pass


def get_image_data(image_data_config_path):
    for image_data_class in image_data_classes:
        try:
            return image_data_class.from_config(image_data_config_path)
        except ValidationError:
            pass
        except TypeError:
            pass
    raise BadDataConfigurationError(
        "The configuration yaml did not match any of the possible image data set configurations."
    )


@click.command()
@click.argument("project-name", type=str)
@click.argument("image-data-config-path", type=click.Path())
@click.option(
    "--env-file",
    type=click.Path(exists=True),
)
def upload_images(project_name, image_data_config_path, env_file):

    if env_file:
        load_dotenv(dotenv_path=env_file)

    image_data = get_image_data(image_data_config_path)
    trainer = get_trainer()
    project_id = get_project_id(trainer, project_name)

    logger.info("Uploading images to Azure.")
    upload_to_custom_vision(image_data, trainer, project_id=project_id)
