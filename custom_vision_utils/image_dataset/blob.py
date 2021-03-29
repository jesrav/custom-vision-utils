import math
from pathlib import Path
from typing import Union, List, Optional, Dict, Iterable

import yaml
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry

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
from custom_vision_utils.image.blob import BlobImage, BlobClassifierImage, BlobObjectDetectionImage
from custom_vision_utils.sdk_helpers.helpers import get_tag_dict, get_tag_id
from custom_vision_utils.image_dataset.image_dataset_interface import ImageDataSetInterface
from custom_vision_utils.pillow_utils import pil_image_to_byte_array


class BlobImageDataSet(ImageDataSetInterface):
    def __init__(self, images: Union[List[BlobImage], None] = None):
        self.images = images

    def append(self, blob_image: BlobImage) -> None:
        if not isinstance(blob_image, BlobImage):
            raise ValueError("classifier_image must be type BlobImage.")
        if not self.images:
            self.images = [blob_image]
        else:
            self.images = self.images + [blob_image]

    def __add__(self, other):
        if not isinstance(other, BlobImageDataSet):
            raise ValueError(
                "You can only add ther objects of type BlobImageData."
            )
        return BlobImageDataSet(images=self.images + other.images)

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        for i in range(len(self.images)):
            yield self.images[i]

    @classmethod
    def _from_dir_config(cls, config: dict, connection_str: Optional[str] = None):
        dir_config = BlobImageDataDirConfig(**config)
        blob_image_data = BlobImageDataSet()
        for image_dir in dir_config.image_dirs:
            container_name = image_dir.container_name
            for blob in list_blobs(
                container_name=container_name,
                name_starts_with=image_dir.blob_dir,
                ends_with=".jpg",
                connection_string=connection_str,
            ):
                blob_image_data.append(
                    BlobImage(
                        uri=blob.name,
                        container_name=container_name,
                        connection_str=connection_str
                    )
                )
        return blob_image_data

    @classmethod
    def _from_flat_config(cls, config: Dict, connection_str: Optional[str] = None):
        flat_config = BlobImageDataFlatConfig(**config)
        blob_image_data = BlobImageDataSet()

        for image in flat_config.images:
            uri = image.uri
            container_name = image.container_name
            blob_image_data.append(
                BlobImage(
                    uri=uri,
                    container_name=container_name,
                    connection_str=connection_str
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
                BlobImageConfig(
                    uri=image.uri,
                    container_name=image.container_name
                )
                for image in self
            ],
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
            azure_images.append(
                ImageFileCreateEntry(
                    name=image.uri,
                    contents=pil_image_to_byte_array(image.get_pil_image()),
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


class BlobClassifierDataSet(ImageDataSetInterface):
    def __init__(self, images: Union[List[BlobClassifierImage], None] = None):
        self.images = images

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
                "You can only add ther objects of type BlobClassifierData."
            )
        return BlobClassifierDataSet(images=self.images + other.images)

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        for i in range(len(self.images)):
            yield self.images[i]

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
                ends_with=".jpg",
                connection_string=connection_str,
            ):
                blob_classifier_data.append(
                    BlobClassifierImage(
                        uri=blob.name,
                        tag_names=tag_names,
                        container_name=container_name,
                        connection_str=connection_str
                    )
                )
        return blob_classifier_data

    @classmethod
    def _from_flat_config(cls, config: Dict, connection_str: Optional[str] = None):
        flat_config = BlobClassifierDataFlatConfig(**config)
        blob_classifier_data = BlobClassifierDataSet()

        for image in flat_config.images:
            uri = image.uri
            tag_names = image.tag_names
            container_name = image.container_name
            blob_classifier_data.append(
                BlobClassifierImage(
                    uri=uri,
                    tag_names=tag_names,
                    container_name=container_name,
                    connection_str=connection_str
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
                    container_name=image.container_name
                )
                for image in self
            ],
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
            azure_images.append(
                ImageFileCreateEntry(
                    name=image.uri,
                    contents=pil_image_to_byte_array(image.get_pil_image()),
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


class BlobObjectDetectionDataSet(ImageDataSetInterface):
    def __init__(self, images: Optional[List[BlobObjectDetectionImage]] = None):
        self.images = images

    def append(self, blob_object_detection_image: BlobObjectDetectionImage) -> None:
        if not isinstance(blob_object_detection_image, BlobObjectDetectionImage):
            raise ValueError("blob_object_detection_image must be type BlobObjectDetectionImage.")
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

    @classmethod
    def _from_flat_config(cls, config: Dict, connection_str: Optional[str] = None):
        flat_config = BlobObjectDetectionDataSet(**config)
        blob_object_detection_data = BlobObjectDetectionDataSet()

        for image in flat_config.images:
            uri = image.uri
            regions = image.regions
            container_name = image.container_name
            blob_object_detection_data.append(
                BlobObjectDetectionImage(
                    uri=uri,
                    regions=regions,
                    container_name=container_name,
                    connection_str=connection_str
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
                    regions=image.regions
                )
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