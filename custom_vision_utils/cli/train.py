import time

import click
from dotenv import load_dotenv, find_dotenv

from custom_vision_utils.sdk_helpers import get_trainer, get_project_id
from logger import logger


def train(trainer, project_id):
    iteration = trainer.train_project(project_id)
    while iteration.status != "Completed":
        iteration = trainer.get_iteration(project_id, iteration.id)
        logger.info("Training status: " + iteration.status)
        time.sleep(10)


@click.command()
@click.argument("project_name", type=str)
def train(project_name):

    # Loading environment variables from .env file
    load_dotenv(find_dotenv())

    trainer = get_trainer()
    project_id = get_project_id(trainer, project_name)
    logger.info(f"Training project: {project_name}")
    breakpoint()
    train(trainer, project_id)

