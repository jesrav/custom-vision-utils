from pathlib import Path
from typing import Optional, List, Dict, Union, Iterable

import yaml

from custom_vision_utils.configurations.local_data import (
    LocalImageDataFlatConfig,
    LocalImageDataDirConfig,
    LocalImageConfig,
    LocalClassifierDataFlatConfig,
    LocalClassifierDataDirConfig,
    LocalClassifierImageConfig,
    LocalObjectDetectionDataFlatConfig, LocalObjectDetectionImageConfig,
)
from custom_vision_utils.image.local import LocalImage, LocalClassifierImage, LocalObjectDetectionImage
from custom_vision_utils.image_dataset.image_dataset_interface import ImageDataSetInterface


class LocalImageDataSet(ImageDataSetInterface):
    def __init__(self, images: Optional[List[LocalImage]] = None):
        self.images = images

    def append(self, local_image: LocalImage) -> None:
        if not isinstance(local_image, LocalImage):
            raise ValueError("classifier_image must be type LocalImage.")
        self.images = [local_image] if not self.images else self.images + [local_image]

    def __add__(self, other: "LocalImageDataSet"):
        if not isinstance(other, LocalImageDataSet):
            raise ValueError(
                "You can only add their objects of type LocalClassifierData."
            )
        return LocalImageDataSet(images=self.images + other.images)

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        for i in range(len(self.images)):
            yield self.images[i]

    def __getitem__(self, item):
        return self.images[item]

    @classmethod
    def _from_flat_config(cls, config: Dict):
        flat_config = LocalImageDataFlatConfig(**config)
        local_classifier_data = LocalImageDataSet()
        for image in flat_config.images:
            uri = image.uri
            local_classifier_data.append(LocalImage(uri=uri))
        return local_classifier_data

    @classmethod
    def _from_dir_config(cls, config):
        dir_config = LocalImageDataDirConfig(**config)
        local_classifier_data = LocalImageDataSet()
        for image_dir in dir_config.image_dirs:
            for path in Path(image_dir.path_dir).glob("*.jpg"):
                local_classifier_data.append(LocalImage(uri=path))
        return local_classifier_data

    @classmethod
    def from_config(cls, yaml_config_path: Union[Path, str]):
        with open(yaml_config_path, "r") as stream:
            config = yaml.safe_load(stream)
        if "images" in config:
            return cls._from_flat_config(config)
        else:
            return cls._from_dir_config(config)

    def get_config(self) -> LocalImageDataFlatConfig:
        return LocalImageDataFlatConfig(
            images=[
                LocalImageConfig(uri=image.uri)
                for image in self
            ]
        )


class LocalClassifierDataSet(ImageDataSetInterface):
    def __init__(self, images: Optional[List[LocalClassifierImage]] = None):
        self.images = images

    def append(self, local_classifier_image: LocalClassifierImage) -> None:
        if not isinstance(local_classifier_image, LocalClassifierImage):
            raise ValueError("local_classifier_image must be type LocalClassifierImage.")
        if not self.images:
            self.images = [local_classifier_image]
        else:
            self.images = self.images + [local_classifier_image]

    def __add__(self, other):
        if not isinstance(other, LocalClassifierDataSet):
            raise ValueError(
                "You can only add ther objects of type LocalClassifierData."
            )
        return LocalClassifierDataSet(images=self.images + other.images)

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        for i in range(len(self.images)):
            yield self.images[i]

    def __getitem__(self, item):
        return self.images[item]

    @classmethod
    def _from_flat_config(cls, config: Dict):
        flat_config = LocalClassifierDataFlatConfig(**config)
        local_classifier_data = LocalClassifierDataSet()
        for image in flat_config.images:
            uri = image.uri
            tag_names = image.tag_names
            local_classifier_data.append(
                LocalClassifierImage(
                    uri=uri,
                    tag_names=tag_names,
                )
            )
        return local_classifier_data

    @classmethod
    def _from_dir_config(cls, config):
        dir_config = LocalClassifierDataDirConfig(**config)
        local_classifier_data = LocalClassifierDataSet()
        for image_dir in dir_config.image_dirs:
            tag_names = image_dir.tag_names
            for path in Path(image_dir.path_dir).glob("*.jpg"):
                local_classifier_data.append(
                    LocalClassifierImage(
                        uri=path,
                        tag_names=tag_names
                    )
                )
        return local_classifier_data

    @classmethod
    def from_config(cls, yaml_config_path: Union[Path, str]):
        with open(yaml_config_path, "r") as stream:
            config = yaml.safe_load(stream)
        if "images" in config:
            return cls._from_flat_config(config)
        else:
            return cls._from_dir_config(config)

    def get_config(self) -> LocalClassifierDataFlatConfig:
        return LocalClassifierDataFlatConfig(
            images = [
                LocalClassifierImageConfig(uri=image.uri, tag_names=image.tag_names)
                for image in self
            ]
        )


class LocalObjectDetectionDataSet(ImageDataSetInterface):
    def __init__(self, images: Optional[List[LocalObjectDetectionImage]] = None):
        self.images = images

    def append(self, local_object_detection_image: LocalObjectDetectionImage) -> None:
        if not isinstance(local_object_detection_image, LocalObjectDetectionImage):
            raise ValueError("local_object_detection_image must be type LocalObjectDetectionImage.")
        if not self.images:
            self.images = [local_object_detection_image]
        else:
            self.images = self.images + [local_object_detection_image]

    def __add__(self, other):
        if not isinstance(other, LocalObjectDetectionDataSet):
            raise ValueError(
                "You can only add ther objects of type LocalClassifierData."
            )
        return LocalObjectDetectionDataSet(images=self.images + other.images)

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        for i in range(len(self.images)):
            yield self.images[i]

    def __getitem__(self, item):
        return self.images[item]

    @classmethod
    def _from_flat_config(cls, config: Dict):
        flat_config = LocalObjectDetectionDataFlatConfig(**config)
        local_object_detection_data = LocalObjectDetectionDataSet()
        for image in flat_config.images:
            uri = image.uri
            regions = image.regions
            local_object_detection_data.append(
                LocalObjectDetectionImage(
                    uri=uri,
                    regions=regions,
                )
            )
        return local_object_detection_data

    @classmethod
    def from_config(cls, yaml_config_path: Union[Path, str]):
        with open(yaml_config_path, "r") as stream:
            config = yaml.safe_load(stream)
        return cls._from_flat_config(config)

    def get_config(self) -> LocalObjectDetectionDataFlatConfig:
        return LocalObjectDetectionDataFlatConfig(
            images = [
                LocalObjectDetectionImageConfig(uri=image.uri, regions=image.regions)
                for image in self
            ]
        )
