from io import BytesIO

from PIL import Image
from imgaug import augmenters as iaa
from numpy import array


def pil_image_to_byte_array(image: Image, format: str = "jpeg") -> bytes:
    """Convert Pillow image to byte array."""
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format=image.format if image.format else format)
    return img_byte_arr.getvalue()


def crop_image(
    image: Image, xmin: float, ymax: float, width: float, height: float
) -> Image:
    """Crops img based on coordinates predicted by the object detection model.

    Parameters
    ----------
    image: Pillow pillow_utils.
    xmin: x minimum for the cropping area. between 0 and 1.
    ymax: y minimum for the cropping area. between 0 and 1.
    width: Width between 0 and 1.
    height: Heigh between 0 and 1.

    Returns
    -------
    Cropped pillow_utils as byte array
    """
    img_width, img_height = image.size

    left = int(xmin * img_width)
    right = int((xmin + width) * img_width)
    top = int(ymax * img_height)
    bottom = int((ymax + height) * img_height)
    img_crop = image.crop((left, top, right, bottom))

    return img_crop


def apply_augmentation_sequence_to_pil_images(
        pil_image: list,
        aug_sequence: iaa.Sequential,
        n_augumented_images: int = 5
) -> list:
    """Apply augmentation sequence to list of pillow images."""
    array_images = [array(pil_image) for _ in range(n_augumented_images)]
    augmented_array_images = aug_sequence(images=array_images)
    return [Image.fromarray(array_image) for array_image in augmented_array_images]
