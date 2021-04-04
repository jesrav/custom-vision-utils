from pathlib import Path
from typing import List, Optional, Union
import yaml

from pydantic import BaseModel, validator

ALLOWED_DOMAINS_NAMES = {
    "Classification": [
        "General",
        "Food",
        "Landmarks",
        "Retail",
        "Adult",
        "General (compact)",
        "Food (compact)",
        "Landmarks (compact)" "Retail (compact)",
    ],
    "ObjectDetection": [
        "General [A1]",
        "General",
        "Logo",
        "Products on Shelves",
        "General (compact)",
        "General (compact) [S1]",
        "General [A1]",
    ],
}
ALLOWED_CLASSIFICATION_TYPES = ["Multiclass", "Multilabel"]
ALLOWED_TAG_TYPES = ["Regular", "Negative", "GeneralProduct"]


class Tag(BaseModel):
    """Class representing a Custum Vision tag."""
    name: str
    type: str

    @validator("type")
    def tag_type_validator(cls, value):
        if value not in ALLOWED_TAG_TYPES:
            raise ValueError(f"tag type must be in {ALLOWED_TAG_TYPES}")
        return value


class CustomVisionProject(BaseModel):
    """Class representing a Custum Vision Project."""
    project_name: str
    domain_type: str
    domain_name: str
    classification_type: Optional[str]
    publish_name: str
    tags: List[Tag]

    @validator("classification_type")
    def classification_type_validator(cls, value):
        if value not in ALLOWED_CLASSIFICATION_TYPES:
            raise ValueError(
                f"classification_type must be in {ALLOWED_CLASSIFICATION_TYPES}"
            )
        return value

    @validator("domain_type")
    def domain_type_validator(cls, value):
        if value not in ALLOWED_DOMAINS_NAMES.keys():
            raise ValueError(f"domain_type must be in {ALLOWED_DOMAINS_NAMES.keys()}")
        return value

    @validator("domain_name")
    def domain_name_validator(cls, value, values):
        allowed = ALLOWED_DOMAINS_NAMES[values["domain_type"]]
        if value not in allowed:
            raise ValueError(
                f"domain_name must be in {allowed} for domain type {values['domain_type']}"
            )
        return value


def load_custom_vision_project_configs(
    custom_vision_config_path: Union[Path, str],
) -> List[CustomVisionProject]:
    with open(custom_vision_config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return [CustomVisionProject(**project) for project in config["projects"]]
