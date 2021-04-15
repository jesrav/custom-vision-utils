from pathlib import Path
from typing import Union, List, Optional, Dict

import yaml

from custom_vision_utils.azure_blob import list_blobs
from custom_vision_utils.configurations.blob_data import (
    BlobImageDataDirConfig,
    BlobImageDataFlatConfig,
    BlobImageConfig,
    BlobClassifierDataDirConfig,
    BlobClassifierDataFlatConfig,
    BlobClassifierImageConfig,
    BlobObjectDetectionImageConfig,
    BlobObjectDetectionDataFlatConfig,
)
from custom_vision_utils.image.blob import (
    BlobImage,
    BlobClassifierImage,
    BlobObjectDetectionImage,
)
from custom_vision_utils.image_dataset.image_dataset_interface import (
    ImageDataSetInterface,
)


class BlobImageDataSet(ImageDataSetInterface):
    def __init__(self, images: Union[List[BlobImage], None] = None):
        self.images = images or []

    def append(self, blob_image: BlobImage) -> None:
        if not isinstance(blob_image, BlobImage):
            raise ValueError("classifier_image must be type BlobImage.")
        self.images = [blob_image] if not self.images else self.images + [blob_image]

    def __add__(self, other):
        if not isinstance(other, BlobImageDataSet):
            raise ValueError("You can only add other objects of type BlobImageData.")
        return BlobImageDataSet(images=self.images + other.images)

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        for i in range(len(self.images)):
            yield self.images[i]

    def __getitem__(self, item):
        return self.images[item]

    @classmethod
    def _from_dir_config(cls, config: dict, connection_str: Optional[str] = None):
        dir_config = BlobImageDataDirConfig(**config)
        blob_image_data = BlobImageDataSet()
        for image_dir in dir_config.image_dirs:
            container_name = image_dir.container_name
            for blob in list_blobs(
                container_name=container_name,
                name_starts_with=image_dir.blob_dir,
                extensions=[".jpg", ".jpeg"],
                connection_string=connection_str,
            ):
                blob_image_data.append(
                    BlobImage(
                        uri=blob.name,
                        container_name=container_name,
                        connection_str=connection_str,
                    )
                )
        return blob_image_data

    @classmethod
    def _from_flat_config(cls, config: Dict, connection_str: Optional[str] = None):
        flat_config = BlobImageDataFlatConfig(**config)
        blob_image_data = BlobImageDataSet()

        for image in flat_config.images:
            blob_image_data.append(
                BlobImage(
                    uri=image.uri,
                    container_name=image.container_name,
                    name=image.name,
                    connection_str=connection_str,
                )
            )
        return blob_image_data

    @classmethod
    def from_config(cls, yaml_config_path: Union[Path, str]):
        with open(yaml_config_path, "r") as stream:
            config = yaml.safe_load(stream)
        if "images" in config:
            return cls._from_flat_config(config)
        else:
            return cls._from_dir_config(config)

    def get_config(self) -> BlobImageDataFlatConfig:
        return BlobImageDataFlatConfig(
            images=[
                BlobImageConfig(uri=image.uri, container_name=image.container_name, name=image.name)
                for image in self
            ],
        )


class BlobClassifierDataSet(ImageDataSetInterface):
    def __init__(self, images: Union[List[BlobClassifierImage], None] = None):
        self.images = images or []

    def append(self, blob_classifier_image: BlobClassifierImage) -> None:
        if not isinstance(blob_classifier_image, BlobClassifierImage):
            raise ValueError("blob_classifier_image must be type BlobClassifierImage.")
        if not self.images:
            self.images = [blob_classifier_image]
        else:
            self.images = self.images + [blob_classifier_image]

    def __add__(self, other):
        if not isinstance(other, BlobClassifierDataSet):
            raise ValueError(
                "You can only add other objects of type BlobClassifierData."
            )
        return BlobClassifierDataSet(images=self.images + other.images)

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        for i in range(len(self.images)):
            yield self.images[i]

    def __getitem__(self, item):
        return self.images[item]

    @classmethod
    def _from_dir_config(cls, config: dict, connection_str: Optional[str] = None):
        dir_config = BlobClassifierDataDirConfig(**config)
        blob_classifier_data = BlobClassifierDataSet()
        for image_dir in dir_config.image_dirs:
            tag_names = image_dir.tag_names
            container_name = image_dir.container_name
            for blob in list_blobs(
                container_name=container_name,
                name_starts_with=image_dir.blob_dir,
                extensions=[".jpg", ".jpeg"],
                connection_string=connection_str,
            ):
                blob_classifier_data.append(
                    BlobClassifierImage(
                        uri=blob.name,
                        tag_names=tag_names,
                        container_name=container_name,
                        connection_str=connection_str,
                    )
                )
        return blob_classifier_data

    @classmethod
    def _from_flat_config(cls, config: Dict, connection_str: Optional[str] = None):
        flat_config = BlobClassifierDataFlatConfig(**config)
        blob_classifier_data = BlobClassifierDataSet()

        for image in flat_config.images:
            blob_classifier_data.append(
                BlobClassifierImage(
                    uri=image.uri,
                    tag_names=image.tag_names,
                    container_name=image.container_name,
                    name=image.name,
                    connection_str=connection_str,
                )
            )
        return blob_classifier_data

    @classmethod
    def from_config(cls, yaml_config_path: Union[Path, str]):
        with open(yaml_config_path, "r") as stream:
            config = yaml.safe_load(stream)
        if "images" in config:
            return cls._from_flat_config(config)
        else:
            return cls._from_dir_config(config)

    def get_config(self) -> BlobClassifierDataFlatConfig:
        return BlobClassifierDataFlatConfig(
            images=[
                BlobClassifierImageConfig(
                    uri=image.uri,
                    tag_names=image.tag_names,
                    container_name=image.container_name,
                    name=image.name,
                )
                for image in self
            ],
        )


class BlobObjectDetectionDataSet(ImageDataSetInterface):
    def __init__(self, images: Optional[List[BlobObjectDetectionImage]] = None):
        self.images = images or []

    def append(self, blob_object_detection_image: BlobObjectDetectionImage) -> None:
        if not isinstance(blob_object_detection_image, BlobObjectDetectionImage):
            raise ValueError(
                "blob_object_detection_image must be type BlobObjectDetectionImage."
            )
        if not self.images:
            self.images = [blob_object_detection_image]
        else:
            self.images = self.images + [blob_object_detection_image]

    def __add__(self, other):
        if not isinstance(other, BlobObjectDetectionDataSet):
            raise ValueError(
                "You can only add objects of type BlobObjectDetectionData."
            )
        return BlobObjectDetectionDataSet(images=self.images + other.images)

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        for i in range(len(self.images)):
            yield self.images[i]

    def __getitem__(self, item):
        return self.images[item]

    @classmethod
    def _from_flat_config(cls, config: Dict, connection_str: Optional[str] = None):
        flat_config = BlobObjectDetectionDataFlatConfig(**config)
        blob_object_detection_data = BlobObjectDetectionDataSet()

        for image in flat_config.images:
            blob_object_detection_data.append(
                BlobObjectDetectionImage(
                    uri=image.uri,
                    regions=image.regions,
                    container_name=image.container_name,
                    name=image.name,
                    connection_str=connection_str,
                )
            )
        return blob_object_detection_data

    @classmethod
    def from_config(cls, yaml_config_path: Union[Path, str]):
        with open(yaml_config_path, "r") as stream:
            config = yaml.safe_load(stream)
        return cls._from_flat_config(config)

    def get_config(self) -> BlobObjectDetectionDataFlatConfig:
        return BlobObjectDetectionDataFlatConfig(
            images=[
                BlobObjectDetectionImageConfig(
                    uri=image.uri,
                    container_name=image.container_name,
                    regions=image.regions,
                    name=image.name,
                )
                for image in self
            ]
        )
