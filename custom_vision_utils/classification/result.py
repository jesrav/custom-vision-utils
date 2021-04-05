from pydantic import BaseModel


class ImageClassifierResult(BaseModel):
    """Class representing the result of a Custum Vision classifier."""
    tag_name: str
    probability: float

    def as_dict(self):
        return {
            "tag_name": self.tag_name,
            "probability": self.probability,
        }
