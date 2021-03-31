import click

from custom_vision_utils.cli.upload_images import upload_images
from custom_vision_utils.cli.create_projects import create_projects
from custom_vision_utils.cli.train import train


@click.group()
def cli():
    pass


cli.add_command(create_projects)
cli.add_command(upload_images)
cli.add_command(train)



if __name__ == "__main__":
    cli()
