from io import BytesIO
from pathlib import Path
from typing import Optional, List

from PIL import Image

from custom_vision_utils.image.image_interface import ImageInterface
from custom_vision_utils.azure_blob import get_blob
from custom_vision_utils.object_detection import Region
from custom_vision_utils.pillow_utils import pil_image_to_byte_array
from custom_vision_utils.sdk_helpers import download_custom_vision_image
from custom_vision_utils.object_detection import BoundingBox

ALLOWED_EXTENSIONS = [".jpeg", ".jpg"]


def _validate_extension(uri: str):
    if not any(uri.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise ValueError(
            f"blob_name must be one of the file types {ALLOWED_EXTENSIONS}."
        )


class BlobImage(ImageInterface):
    def __init__(
        self,
        uri: str,
        container_name: str,
        connection_str: Optional[str] = None,
        name: Optional[str] = None,
    ):
        _validate_extension(uri)
        self.uri = uri
        self.container_name = container_name
        self.name = Path(self.uri).stem if not name else name
        self.blob = get_blob(
            container_name=container_name,
            blob_name=self.uri,
            connection_string=connection_str,
        )

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
        _validate_extension(uri)
        blob = get_blob(container_name, uri, connection_str)
        blob.upload_blob(pil_image_to_byte_array(image), overwrite=overwrite)
        return BlobImage(
            uri=uri,
            container_name=container_name,
            connection_str=connection_str,
            name=name,
        )

    @staticmethod
    def from_azure_custom_vision_image(
        custom_vision_image,
        folder: str,
        container_name: str,
        connection_str: Optional[str] = None,
        overwrite: bool = True,
    ) -> "BlobImage":
        if custom_vision_image.metadata and "name" in custom_vision_image.metadata:
            name = custom_vision_image.metadata["name"]
            blob_uri = folder + f"/{name}" + ".jpg"
        else:
            name = None
            blob_uri = folder + f"/{custom_vision_image.id}" + ".jpg"

        handler = BytesIO()
        download_custom_vision_image(
            custom_vision_image=custom_vision_image, file_handler=handler
        )
        return BlobImage.from_pil_image(
            image=Image.open(handler),
            uri=blob_uri,
            container_name=container_name,
            name=name,
            connection_str=connection_str,
            overwrite=overwrite,
        )


class BlobClassifierImage(ImageInterface):
    def __init__(
        self,
        uri: str,
        tag_names: List[str],
        container_name: str,
        name: Optional[str] = None,
        connection_str: Optional[str] = None,
    ):
        _validate_extension(uri)
        self.uri = uri
        self.container_name = container_name
        self.tag_names = tag_names
        self.name = Path(self.uri).stem if not name else name
        self.blob = get_blob(
            container_name=container_name,
            blob_name=self.uri,
            connection_string=connection_str,
        )

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
        _validate_extension(uri)
        blob = get_blob(container_name, uri, connection_str)
        blob.upload_blob(pil_image_to_byte_array(image), overwrite=overwrite)
        return BlobClassifierImage(
            uri=uri,
            tag_names=tag_names,
            container_name=container_name,
            name=name,
            connection_str=connection_str,
        )

    @staticmethod
    def from_azure_custom_vision_image(
        custom_vision_image,
        folder: str,
        container_name: str,
        connection_str: Optional[str] = None,
        overwrite: bool = True,
    ) -> "BlobClassifierImage":
        if custom_vision_image.metadata and "name" in custom_vision_image.metadata:
            name = custom_vision_image.metadata["name"]
            blob_uri = folder + f"/{name}" + ".jpg"
        else:
            name = None
            blob_uri = folder + f"/{custom_vision_image.id}" + ".jpg"

        handler = BytesIO()
        download_custom_vision_image(
            custom_vision_image=custom_vision_image, file_handler=handler
        )
        return BlobClassifierImage.from_pil_image(
            image=Image.open(handler),
            uri=blob_uri,
            tag_names=[tag.tag_name for tag in custom_vision_image.tags],
            container_name=container_name,
            name=name,
            connection_str=connection_str,
            overwrite=overwrite,
        )


class BlobObjectDetectionImage(ImageInterface):
    def __init__(
        self,
        uri: str,
        regions: List[Region],
        container_name: str,
        name: Optional[str] = None,
        connection_str: Optional[str] = None,
    ):
        _validate_extension(uri)
        self.uri = uri
        self.container_name = container_name
        self.regions = regions
        self.name = Path(self.uri).stem if not name else name
        self.blob = get_blob(
            container_name=container_name,
            blob_name=self.uri,
            connection_string=connection_str,
        )

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
        _validate_extension(uri)
        blob = get_blob(container_name, uri, connection_str)
        blob.upload_blob(pil_image_to_byte_array(image), overwrite=overwrite)
        return BlobObjectDetectionImage(
            uri=uri,
            regions=regions,
            container_name=container_name,
            name=name,
            connection_str=connection_str,
        )

    @staticmethod
    def from_azure_custom_vision_image(
        custom_vision_image,
        folder: str,
        container_name: str,
        connection_str: Optional[str] = None,
        overwrite: bool = True,
    ) -> "BlobObjectDetectionImage":
        if custom_vision_image.metadata and "name" in custom_vision_image.metadata:
            name = custom_vision_image.metadata["name"]
            blob_uri = folder + f"/{name}" + ".jpg"
        else:
            name = None
            blob_uri = folder + f"/{custom_vision_image.id}" + ".jpg"

        handler = BytesIO()
        download_custom_vision_image(
            custom_vision_image=custom_vision_image, file_handler=handler
        )
        return BlobObjectDetectionImage.from_pil_image(
            image=Image.open(handler),
            uri=blob_uri,
            regions=[
                Region(
                    bounding_box=BoundingBox(
                        left=region.left,
                        top=region.top,
                        width=region.width,
                        height=region.height,
                    ),
                    tag_name=region.tag_name,
                )
                for region in custom_vision_image.regions
            ],
            container_name=container_name,
            name=name,
            connection_str=connection_str,
            overwrite=overwrite,
        )
