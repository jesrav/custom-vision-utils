from io import BytesIO
from pathlib import Path
from typing import Optional, List

from PIL import Image

from custom_vision_utils.image.image_interface import ImageInterface
from custom_vision_utils.azure_blob import get_blob
from custom_vision_utils.object_detection import Region
from custom_vision_utils.pillow_utils import pil_image_to_byte_array


class BlobImage(ImageInterface):
    def __init__(
            self,
            uri: str,
            container_name: str,
            connection_str: Optional[str] = None,
            name: Optional[str] = None
    ):
        if not uri.endswith(".jpg"):
            raise ValueError("uri must be of file type jpg.")
        self.uri = uri
        self.container_name = container_name
        self.name = Path(self.uri).stem if not name else name
        self.blob = get_blob(container_name=container_name, blob_name=self.uri,connection_string=connection_str)

    def get_pil_image(self) -> Image:
        handler = BytesIO()
        blob_data = self.blob.download_blob()
        handler.write(blob_data.readall())
        return Image.open(handler)

    @staticmethod
    def from_pil_image(
            image: Image,
            uri: str,
            container_name: str,
            name: Optional[str] = None,
            connection_str: Optional[str] = None,
            overwrite: bool = True,
    ) -> "BlobImage":
        if not uri.endswith(".jpg"):
            raise ValueError("blob_name must be of file type jpg.")
        blob = get_blob(container_name, uri, connection_str)
        blob.upload_blob(pil_image_to_byte_array(image), overwrite=overwrite)
        return BlobImage(
            uri=uri,
            container_name=container_name,
            connection_str=connection_str,
            name=name
        )


class BlobClassifierImage(ImageInterface):
    def __init__(
            self,
            uri: str,
            tag_names: List[str],
            container_name: str,
            name: Optional[str] = None,
            connection_str: Optional[str] = None
    ):
        if not uri.endswith(".jpg"):
            raise ValueError("uri must be of file type jpg.")
        self.uri = uri
        self.container_name = container_name
        self.tag_names = tag_names
        self.name = Path(self.uri).stem if not name else name
        self.blob = get_blob(container_name=container_name, blob_name=self.uri, connection_string=connection_str)

    def get_pil_image(self) -> Image:
        handler = BytesIO()
        blob_data = self.blob.download_blob()
        handler.write(blob_data.readall())
        return Image.open(handler)

    @staticmethod
    def from_pil_image(
            image: Image,
            uri: str,
            tag_names: List[str],
            container_name: str,
            name: Optional[str] = None,
            connection_str: Optional[str] = None,
            overwrite: bool = True,
    ) -> "BlobClassifierImage":
        if not uri.endswith(".jpg"):
            raise ValueError("blob_name must be of file type jpg.")
        blob = get_blob(container_name, uri, connection_str)
        blob.upload_blob(pil_image_to_byte_array(image), overwrite=overwrite)
        return BlobClassifierImage(
            uri=uri,
            tag_names=tag_names,
            container_name=container_name,
            name=name,
            connection_str=connection_str
        )


class BlobObjectDetectionImage(ImageInterface):
    def __init__(
            self,
            uri: str,
            regions: List[Region],
            container_name: str,
            name: Optional[str] = None,
            connection_str: Optional[str] = None
    ):
        if not uri.endswith(".jpg"):
            raise ValueError("uri must be of file type jpg.")
        self.uri = uri
        self.container_name = container_name
        self.regions = regions
        self.name = Path(self.uri).stem if not name else name
        self.blob = get_blob(container_name=container_name, blob_name=self.uri, connection_string=connection_str)

    def get_pil_image(self) -> Image:
        handler = BytesIO()
        blob_data = self.blob.download_blob()
        handler.write(blob_data.readall())
        return Image.open(handler)

    @staticmethod
    def from_pil_image(
            image: Image,
            uri: str,
            regions: List[Region],
            container_name: str,
            name: Optional[str] = None,
            connection_str: Optional[str] = None,
            overwrite: bool = True,
    ) -> "BlobObjectDetectionImage":
        if not uri.endswith(".jpg"):
            raise ValueError("blob_name must be of file type jpg.")
        blob = get_blob(container_name, uri, connection_str)
        blob.upload_blob(pil_image_to_byte_array(image), overwrite=overwrite)
        return BlobObjectDetectionImage(
            uri=uri,
            regions=regions,
            container_name=container_name,
            name=name,
            connection_str=connection_str
        )