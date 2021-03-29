import math
from pathlib import Path
from typing import Optional, List, Dict, Union, Iterable

import yaml

from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry
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
from custom_vision_utils.sdk_helpers.helpers import get_tag_dict, get_tag_id
from custom_vision_utils.image_dataset.image_dataset_interface import ImageDataSetInterface


class LocalImageDataSet(ImageDataSetInterface):
    def __init__(self, images: Optional[List[LocalImage]] = None):
        self.images = images

    def append(self, local_image: LocalImage) -> None:
        if not isinstance(local_image, LocalImage):
            raise ValueError("classifier_image must be type LocalImage.")
        if not self.images:
            self.images = [local_image]
        else:
            self.images = self.images + [local_image]

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

    def _get_azure_images(
        self,
        start_inx,
        stop_inx,
    ) -> List[ImageFileCreateEntry]:
        """Get of images in Azure specific format.

        :param trainer: CustomVisionTrainingClient
        :param project_id: Custom Vision project id
        :param start_index: Start index of images to generate from self.images
        :param stop_inx: Start index of images to generate from self.images
        :return:
        """
        if not start_inx:
            start_inx = 0
        if not stop_inx:
            stop_inx = len(self.images)

        azure_images = []
        for image in self.images[start_inx:stop_inx]:
            with open(image.uri, "rb") as image_contents:
                azure_images.append(
                    ImageFileCreateEntry(
                        name=image.uri.stem,
                        contents=image_contents.read()
                    )
                )
        return azure_images

    def get_azure_image_batches(
        self, batch_size=64
    ) -> Iterable[List[ImageFileCreateEntry]]:
        """
        creates a generator that yields batches of images in Azure specific format of size batch_size. Custom Vision
        can maximally upload batches of 64.
        """
        num_batches = math.ceil(len(self.images) / batch_size)
        for i in range(num_batches):
            start_idx = i * batch_size
            stop_inx = start_idx + batch_size
            yield self._get_azure_images(
                start_inx=start_idx,
                stop_inx=stop_inx,
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

    def _get_azure_images(
        self,
        trainer,
        project_id,
        start_inx,
        stop_inx,
    ) -> List[ImageFileCreateEntry]:
        """Get of images in Azure specific format.

        :param trainer: CustomVisionTrainingClient
        :param project_id: Custom Vision project id
        :param start_index: Start index of images to generate from self.images
        :param stop_inx: Start index of images to generate from self.images
        :return:
        """
        if not start_inx:
            start_inx = 0
        if not stop_inx:
            stop_inx = len(self.images)

        project_tag_dict = get_tag_dict(trainer, project_id)
        azure_images = []
        for image in self.images[start_inx:stop_inx]:
            with open(image.uri, "rb") as image_contents:
                azure_images.append(
                    ImageFileCreateEntry(
                        name=image.uri.stem,
                        contents=image_contents.read(),
                        tag_ids=[
                            get_tag_id(project_tag_dict, tag)
                            for tag in image.tag_names
                        ],
                    )
                )
        return azure_images

    def get_azure_image_batches(
        self, trainer, project_id, batch_size=64
    ) -> Iterable[List[ImageFileCreateEntry]]:
        """
        creates a generator that yields batches of images in Azure specific format of size batch_size. Custom Vision
        can maximally upload batches of 64.
        """
        num_batches = math.ceil(len(self.images) / batch_size)
        for i in range(num_batches):
            start_idx = i * batch_size
            stop_inx = start_idx + batch_size
            yield self._get_azure_images(
                trainer=trainer,
                project_id=project_id,
                start_inx=start_idx,
                stop_inx=stop_inx,
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

    def _get_azure_images(
        self,
        trainer,
        project_id,
        start_inx,
        stop_inx,
    ) -> List[ImageFileCreateEntry]:
        """Get of images in Azure specific format.

        :param trainer: CustomVisionTrainingClient
        :param project_id: Custom Vision project id
        :param start_index: Start index of images to generate from self.images
        :param stop_inx: Start index of images to generate from self.images
        :return:
        """
        if not start_inx:
            start_inx = 0
        if not stop_inx:
            stop_inx = len(self.images)

        project_tag_dict = get_tag_dict(trainer, project_id)
        azure_images = []
        for image in self.images[start_inx:stop_inx]:
            with open(image.uri, "rb") as image_contents:
                azure_images.append(
                    ImageFileCreateEntry(
                        name=image.uri.stem,
                        contents=image_contents.read(),
                        regions=[
                            region.to_azure_region(
                                tag_id=project_tag_dict[region.tag_name]
                            )
                            for region in image.regions
                        ],
                    )
                )
        return azure_images

    def get_azure_image_batches(
        self, trainer, project_id, batch_size=64
    ) -> Iterable[List[ImageFileCreateEntry]]:
        """
        creates a generator that yields batches of images in Azure specific format of size batch_size. Custom Vision
        can maximally upload batches of 64.
        """
        num_batches = math.ceil(len(self.images) / batch_size)
        for i in range(num_batches):
            start_idx = i * batch_size
            stop_inx = start_idx + batch_size
            yield self._get_azure_images(
                trainer=trainer,
                project_id=project_id,
                start_inx=start_idx,
                stop_inx=stop_inx,
            )