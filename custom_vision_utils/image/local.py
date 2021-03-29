from pathlib import Path
from typing import Union, List, Optional

import requests
from PIL import Image
from pydantic import validator

from custom_vision_utils.object_detection import Region
from custom_vision_utils.image.image_interface import ImageInterface
from custom_vision_utils.object_detection import BoundingBox


def _download_image(url, outpath: Path):
    r = requests.get(url, allow_redirects=True)
    open(outpath, "wb").write(r.content)


class LocalImage(ImageInterface):

    def __init__(self, uri: Union[str, Path], name: Optional[str] = None) -> None:
        if type(uri) == str:
            uri = Path(uri)
        if not uri.exists():
            raise ValueError(f"Image file {uri} does not exist.")
        self.uri = uri
        self.name = self.uri.stem if not name else name

    @staticmethod
    @validator("uri")
    def _validator_file_format(value):
        if value.suffix != ".jpg":
            raise ValueError("filepath must be .jpg")
        return value

    def get_pil_image(self) -> Image:
        return Image.open(self.uri)

    @staticmethod
    def from_pil_image(image: Image, uri: Path, name: Optional[str] = None) -> "LocalImage":
        image.save(uri)
        return LocalImage(uri=uri, name=name)

    @staticmethod
    def from_azure_custom_vision_image(
            custom_vision_image,
            folder: Path
    ) -> "LocalImage":
        local_image_path = folder / Path(custom_vision_image.id + ".jpg")
        _download_image(
            url=custom_vision_image.original_image_uri,
            outpath=local_image_path
        )
        return LocalImage(uri=local_image_path, name=None)


class LocalClassifierImage(ImageInterface):
    def __init__(self, uri: Union[str, Path], tag_names: List[str], name: Optional[str] = None) -> None:
        if type(uri) == str:
            uri = Path(uri)
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
        uri: Path,
        tag_names: List[str],
        name: Optional[str] = None
    ) -> "LocalClassifierImage":
        image.save(uri)
        return LocalClassifierImage(uri=uri, tag_names=tag_names, name=name)

    @staticmethod
    def from_azure_custom_vision_image(
            custom_vision_image,
            folder: Path
    ) -> "LocalClassifierImage":
        local_image_path = folder / Path(custom_vision_image.id + ".jpg")
        _download_image(
            url=custom_vision_image.original_image_uri,
            outpath=local_image_path
        )
        return LocalClassifierImage(
            uri=local_image_path,
            tag_names=[tag.tag_name for tag in custom_vision_image.tags],
            name=None
        )


class LocalObjectDetectionImage(ImageInterface):
    def __init__(self, uri: Union[str, Path], regions: List[Region], name: Optional[str] = None) -> None:
        if type(uri) == str:
            uri = Path(uri)
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
        uri: Path,
        regions: List[Region],
        name: Optional[str] = None,
    ) -> "LocalObjectDetectionImage":
        image.save(uri)
        return LocalObjectDetectionImage(uri=uri, regions=regions, name=name)

    @staticmethod
    def from_azure_custom_vision_image(
            custom_vision_image,
            folder: Path
    ) -> "LocalObjectDetectionImage":
        local_image_path = folder / Path(custom_vision_image.id + ".jpg")
        _download_image(
            url=custom_vision_image.original_image_uri,
            outpath=local_image_path
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
                ) for region in custom_vision_image.regions]
        )