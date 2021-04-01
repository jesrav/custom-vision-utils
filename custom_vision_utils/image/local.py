from pathlib import Path
from typing import Union, List, Optional

from PIL import Image

from custom_vision_utils.object_detection import Region
from custom_vision_utils.image.image_interface import ImageInterface
from custom_vision_utils.object_detection import BoundingBox
from custom_vision_utils.sdk_helpers import download_custom_vision_image

ALLOWED_EXTENSIONS = [".jpeg", ".jpg"]


class LocalImage(ImageInterface):

    def __init__(self, uri: Union[str, Path], name: Optional[str] = None) -> None:
        uri = Path(uri)
        if uri.suffix not in ALLOWED_EXTENSIONS:
            raise ValueError(f"filepath extension must be one of {ALLOWED_EXTENSIONS}")
        if not uri.exists():
            raise ValueError(f"Image file {uri} does not exist.")

        self.uri = uri
        self.name = self.uri.stem if not name else name

    def get_pil_image(self) -> Image:
        return Image.open(self.uri)

    @staticmethod
    def from_pil_image(image: Image, uri: Union[Path, str], name: Optional[str] = None) -> "LocalImage":
        image.save(uri)
        return LocalImage(uri=uri, name=name)

    @staticmethod
    def from_azure_custom_vision_image(
            custom_vision_image,
            folder: Union[Path, str],
            container=None,
    ) -> "LocalImage":
        _ = container
        folder = Path(folder)

        if custom_vision_image.metadata and 'name' in custom_vision_image.metadata:
            name = custom_vision_image.metadata['name']
            local_image_path = folder / Path(name + ".jpg")
        else:
            name = None
            local_image_path = folder / Path(custom_vision_image.id + ".jpg")

        with open(local_image_path, "wb") as f:
            download_custom_vision_image(
                custom_vision_image=custom_vision_image,
                file_handler=f
            )
        return LocalImage(uri=local_image_path, name=name)


class LocalClassifierImage(ImageInterface):
    def __init__(self, uri: Union[str, Path], tag_names: List[str], name: Optional[str] = None) -> None:
        uri = Path(uri)
        if uri.suffix not in ALLOWED_EXTENSIONS:
            raise ValueError(f"filepath extension must be one of {ALLOWED_EXTENSIONS}")
        if not uri.exists():
            raise ValueError(f"Image file {uri} does not exist.")
        self.uri = uri
        self.name = self.uri.stem if not name else name
        self.tag_names = tag_names

    def get_pil_image(self) -> Image:
        return Image.open(self.uri)

    @staticmethod
    def from_pil_image(
        image: Image,
        uri: Union[Path, str],
        tag_names: List[str],
        name: Optional[str] = None
    ) -> "LocalClassifierImage":
        image.save(uri)
        return LocalClassifierImage(uri=uri, tag_names=tag_names, name=name)

    @staticmethod
    def from_azure_custom_vision_image(
            custom_vision_image,
            folder: Union[Path, str],
            container=None,
    ) -> "LocalClassifierImage":
        _ = container
        folder = Path(folder)

        if custom_vision_image.metadata and 'name' in custom_vision_image.metadata:
            name = custom_vision_image.metadata['name']
            local_image_path = folder / Path(name + ".jpg")
        else:
            name = None
            local_image_path = folder / Path(custom_vision_image.id + ".jpg")

        with open(local_image_path, "wb") as f:
            download_custom_vision_image(
                custom_vision_image=custom_vision_image,
                file_handler=f
            )
        return LocalClassifierImage(
            uri=local_image_path,
            tag_names=[tag.tag_name for tag in custom_vision_image.tags],
            name=name
        )


class LocalObjectDetectionImage(ImageInterface):
    def __init__(self, uri: Union[str, Path], regions: List[Region], name: Optional[str] = None) -> None:
        uri = Path(uri)
        if uri.suffix not in ALLOWED_EXTENSIONS:
            raise ValueError(f"filepath extension must be one of {ALLOWED_EXTENSIONS}")
        if not uri.exists():
            raise ValueError(f"Image file {uri} does not exist.")
        self.uri = uri
        self.name = self.uri.stem if not name else name
        self.regions = regions

    def get_pil_image(self) -> Image:
        return Image.open(self.uri)

    @staticmethod
    def from_pil_image(
        image: Image,
        uri: Union[Path, str],
        regions: List[Region],
        name: Optional[str] = None,
    ) -> "LocalObjectDetectionImage":
        image.save(uri)
        return LocalObjectDetectionImage(uri=uri, regions=regions, name=name)

    @staticmethod
    def from_azure_custom_vision_image(
            custom_vision_image,
            folder: Union[Path, str],
            container=None,
    ) -> "LocalObjectDetectionImage":
        _ = container
        folder = Path(folder)

        if custom_vision_image.metadata and 'name' in custom_vision_image.metadata:
            name = custom_vision_image.metadata['name']
            local_image_path = folder / Path(name + ".jpg")
        else:
            name = None
            local_image_path = folder / Path(custom_vision_image.id + ".jpg")

        with open(local_image_path, "wb") as f:
            download_custom_vision_image(
                custom_vision_image=custom_vision_image,
                file_handler=f
            )
        return LocalObjectDetectionImage(
            uri=local_image_path,
            regions=[
                Region(
                    bounding_box=BoundingBox(
                        left=region.left,
                        top=region.top,
                        width=region.width,
                        height=region.height,
                    ),
                    tag_name=region.tag_name
                ) for region in custom_vision_image.regions],
            name=name
        )

