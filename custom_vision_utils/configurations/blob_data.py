"""Configuration classes for image data sets stored in Azure blob storage."""
from typing import List

from pydantic import BaseModel

from custom_vision_utils.object_detection import Region

####################################################
# Configuration classes for image data sets,
# with no metadata on tag names, regions, etc.
####################################################


class BlobImageConfig(BaseModel):
    """Class representing a single image in blob storage."""
    uri: str
    container_name: str


class BlobImageDataFlatConfig(BaseModel):
    """Class representing a list of images in blob storage.

    The set of images make up the configuration of the image data set.
    """
    images: List[BlobImageConfig]


class BlobImageDir(BaseModel):
    """Class representing a directory of images in blob storage"""
    blob_dir: str
    container_name: str


class BlobImageDataDirConfig(BaseModel):
    """Class representing a list of directories in blob storage.

    The set of images  included in the directories make up the configuration of the image data set.
    """
    image_dirs: List[BlobImageDir]


####################################################
# Configuration classes for image data sets
# for classification, including metadata on tag names.
####################################################
class BlobClassifierImageConfig(BlobImageConfig, BaseModel):
    """Class representing a single classifier image in blob storage."""
    tag_names: List[str]


class BlobClassifierDataFlatConfig(BaseModel):
    """Class representing a list of images in blob storage,
    including tags for classification.

    The set of images make up the configuration of the image data set.
    """
    images: List[BlobClassifierImageConfig]


class BlobClassifierImageDir(BlobImageDir, BaseModel):
    """Class representing a directory images, that all have the same tags, in blob storage"""
    tag_names: List[str]


class BlobClassifierDataDirConfig(BaseModel):
    """Class representing a list of directories in blob storage, with images that have the same tag_name.

    The set of images and accompanying tags included in the directories make up the configuration of the
    classification image data set.
    """
    image_dirs: List[BlobClassifierImageDir]


####################################################
# Configuration classes for image data sets
# for object detection, including metadata on regions.
####################################################
class BlobObjectDetectionImageConfig(BlobImageConfig, BaseModel):
    """Class representing a single object detection image in blob storage,
    including regions with tags.
    """
    regions: List[Region]


class BlobObjectDetectionDataFlatConfig(BaseModel):
    """Class representing a list of images in blob storage,
    including regions with tags.

    The set of images make up the configuration of the image data set.
    """
    images: List[BlobObjectDetectionImageConfig]
