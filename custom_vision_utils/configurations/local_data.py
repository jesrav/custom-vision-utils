"""Configuration classes for image data sets stored locally."""
from typing import List

from pydantic import BaseModel, DirectoryPath, FilePath

from custom_vision_utils.object_detection import Region

####################################################
# Configuration classes for image data sets,
# with no metadata on tag names, regions, etc.
####################################################


class LocalImageConfig(BaseModel):
    """Class representing a single image in local storage."""

    uri: FilePath


class LocalImageDataFlatConfig(BaseModel):
    """Class representing a list of images in local storage.

    The set of images make up the configuration of the image data set.
    """

    images: List[LocalImageConfig]


class LocalImageDir(BaseModel):
    """Class representing a directory of images in local storage"""

    path_dir: DirectoryPath


class LocalImageDataDirConfig(BaseModel):
    """Class representing a list of directories in local storage.

    The set of images  included in the directories make up the configuration of the image data set.
    """

    image_dirs: List[LocalImageDir]


####################################################
# Configuration classes for image data sets
# for classification, including metadata on tag names.
####################################################
class LocalClassifierImageConfig(LocalImageConfig, BaseModel):
    """Class representing a single classifier image in local storage."""

    tag_names: List[str]


class LocalClassifierDataFlatConfig(BaseModel):
    """Class representing a list of images in local storage,
    including tags for classification.

    The set of images make up the configuration of the image data set.
    """

    images: List[LocalClassifierImageConfig]


class LocalClassifierImageDir(LocalImageDir, BaseModel):
    """Class representing a directory images, that all have the same tags, in local storage"""

    tag_names: List[str]


class LocalClassifierDataDirConfig(BaseModel):
    """Class representing a list of directories in local storage, with images that have the same tag_name.

    The set of images and accompanying tags included in the directories make up the configuration of the
    classification image data set.
    """

    image_dirs: List[LocalClassifierImageDir]


####################################################
# Configuration classes for image data sets
# for object detection, including metadata on regions.
####################################################
class LocalObjectDetectionImageConfig(LocalImageConfig, BaseModel):
    """Class representing a single object detection image in local storage,
    including regions with tags.
    """

    regions: List[Region]


class LocalObjectDetectionDataFlatConfig(BaseModel):
    """Class representing a list of images in local storage,
    including regions with tags.

    The set of images make up the configuration of the image data set.
    """

    images: List[LocalObjectDetectionImageConfig]
