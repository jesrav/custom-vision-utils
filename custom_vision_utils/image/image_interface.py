from abc import ABC, abstractmethod


class ImageInterface(ABC):

    @abstractmethod
    def get_pil_image(self):
        pass



