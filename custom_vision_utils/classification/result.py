from pydantic import BaseModel


class ImageClassifierResult(BaseModel):
    tag_name: str
    probability: float

    def as_dict(self):
        return {
            "tag_name": self.tag_name,
            "probability": self.probability,
        }
