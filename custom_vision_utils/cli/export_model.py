from pathlib import Path

import click
from dotenv import load_dotenv, find_dotenv

from custom_vision_utils.sdk_helpers import (
    get_trainer,
    download_model_iteration_as_tensorflow,
    get_iteration,
)
from custom_vision_utils.cli.logger import logger


@click.command()
@click.argument("project_name", type=str)
@click.argument("outpath", type=str)
@click.option("--iteration_name", type=str)
def export_model(project_name, outpath, iteration_name):

    # Loading environment variables from .env file
    load_dotenv(find_dotenv())

    trainer = get_trainer()
    if iteration_name:
        iteration_id = get_iteration(trainer, project_name, iteration_name).id
    else:
        iteration_id = None
        iteration_name = "latest"

    logger.info(
        f"Downloading model artifacts for project {project_name}, "
        f"iteration: {iteration_name} to {outpath}"
    )

    download_model_iteration_as_tensorflow(project_name, Path(outpath), iteration_id)
