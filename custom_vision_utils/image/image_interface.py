from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union


class ImageInterface(ABC):

    @abstractmethod
    def get_pil_image(self):
        pass

    @staticmethod
    @abstractmethod
    def from_azure_custom_vision_image(
            custom_vision_image,
            folder: Union[Path, str],
            container
    ):
        pass
