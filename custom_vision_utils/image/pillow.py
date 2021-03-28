# import uuid
# from typing import List
#
# from PIL import Image
# from .image_interface import ImageInterface
# from custom_vision_utils.object_detection import Region
#
#
# class PillowImage(ImageInterface):
#
#     def __init__(self, image: Image) -> None:
#         self.image = image
#         self.generated_name = str(uuid.uuid4())
#
#     @property
#     def name(self):
#         return self.generated_name
#
#     def get_pil_image(self) -> Image:
#         return self.image
#
#     @staticmethod
#     def from_pil_image(image: Image) -> "PillowImage":
#         return  PillowImage(image=image)
#
#
# class PillowClassifierImage(PillowImage):
#     def __init__(self, image, tage_names: List[str]):
#         super().__init__(image=image)
#         self.tag_names: tage_names
#
#
# class PillowObjectDetectionImage(PillowImage):
#     def __init__(self, image: Image, regions = List[Region]):
#         super().__init__(image=image)
#         self.regions = regions
