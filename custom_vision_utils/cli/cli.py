import click

from custom_vision_utils.cli.upload_images import upload_images
from custom_vision_utils.cli.create_projects import create_projects
from custom_vision_utils.cli.train import train
from custom_vision_utils.cli.export_model import export_model
from custom_vision_utils.cli.export_images import export_images


@click.group()
def cli():
    pass


cli.add_command(create_projects)
cli.add_command(upload_images)
cli.add_command(train)
cli.add_command(export_model)
cli.add_command(export_images)


if __name__ == "__main__":
    cli()
