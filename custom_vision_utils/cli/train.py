import time

import click
from dotenv import load_dotenv, find_dotenv

from custom_vision_utils.sdk_helpers import get_trainer, get_project_id
from custom_vision_utils.cli.logger import logger


def train_project(trainer, project_id):
    iteration = trainer.train_project(project_id)
    while iteration.status != "Completed":
        iteration = trainer.get_iteration(project_id, iteration.id)
        logger.info("Training status: " + iteration.status)
        time.sleep(10)


@click.command()
@click.argument("project_name", type=str)
@click.option(
    "--env-file",
    type=click.Path(exists=True),
)
def train(project_name, env_file):

    if env_file:
        load_dotenv(dotenv_path=env_file)

    trainer = get_trainer()
    project_id = get_project_id(trainer, project_name)
    logger.info(f"Training project: {project_name}")
    train_project(trainer, project_id)
