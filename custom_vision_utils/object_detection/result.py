from azure.cognitiveservices.vision.customvision.training.models import (
    Region as AzureRegion,
)
from pydantic import BaseModel


class BoundingBox(BaseModel):
    left: float
    top: float
    width: float
    height: float

    def as_dict(self):
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


class Region(BaseModel):
    bounding_box: BoundingBox
    tag_name: str

    def to_azure_region(self, tag_id):
        return AzureRegion(
            tag_id=tag_id,
            left=self.bounding_box.left,
            top=self.bounding_box.top,
            width=self.bounding_box.width,
            height=self.bounding_box.height,
        )


class ObjectDetectionResult(BaseModel):
    tag_name: str
    probability: float
    bounding_box: BoundingBox

    def as_dict(self):
        return {
            "tag_name": self.tag_name,
            "probability": self.probability,
            "bounding_box": self.bounding_box.as_dict(),
        }
