from pathlib import Path
from typing import List

import click
from dotenv import load_dotenv

from custom_vision_utils.sdk_helpers import (
    get_trainer,
    get_domain_id,
    get_project_names,
    get_project_id,
)
from custom_vision_utils.configurations.custom_vision_projects import (
    load_custom_vision_project_configs,
    Tag,
)
from custom_vision_utils.cli.logger import logger


def create_project(
    project_name, project_domain_name, project_domain_type, classification_type=None
):
    trainer = get_trainer()
    if project_name not in get_project_names(trainer=trainer):
        if classification_type:
            logger.info(
                f"Creating project {project_name} of type {project_domain_type}, domain name {project_domain_name} "
                f"and classification type {classification_type}."
            )
        else:
            logger.info(
                f"Creating project {project_name} of type {project_domain_type} and domain name {project_domain_name}."
            )
        _ = trainer.create_project(
            project_name,
            domain_id=get_domain_id(trainer, project_domain_name, project_domain_type),
            classification_type=classification_type,
        )
    else:
        logger.info(f"Not creating project. Project {project_name} already created.")


def create_tags(project_id: str, tags: List[Tag]):
    trainer = get_trainer()

    already_created_tags_names = [tag.name for tag in trainer.get_tags(project_id)]
    for tag in tags:
        if tag.name in already_created_tags_names:
            logger.info(f"Tag {tag.name} already created.")
        else:
            logger.info(f"Creating tag {tag.name} of type {tag.type}.")
            _ = trainer.create_tag(project_id, name=tag.name, type=tag.type)


@click.command()
@click.argument(
    "projects-config-path",
    type=click.Path(exists=True),
)
@click.option(
    "--env-file",
    type=click.Path(exists=True),
)
def create_projects(projects_config_path, env_file):

    if env_file:
        load_dotenv(dotenv_path=env_file)

    projects_config_path = Path(projects_config_path)

    if Path(projects_config_path).suffix != ".yaml":
        raise ValueError("File extension of `projects-config-path` must be `.yaml`.")
    custom_vision_configs = load_custom_vision_project_configs(
        Path(projects_config_path)
    )

    for cv_config in custom_vision_configs:
        create_project(
            project_name=cv_config.project_name,
            project_domain_name=cv_config.domain_name,
            project_domain_type=cv_config.domain_type,
            classification_type=cv_config.classification_type,
        )
        create_tags(
            project_id=get_project_id(
                get_trainer(),
                project_name=cv_config.project_name,
            ),
            tags=cv_config.tags,
        )
