import hashlib
import uuid
from PIL import Image


def get_hash_from_uuid(hash_len: int = 5) -> str:
    # Generate a UUID4 and convert it to a string
    uuid_str = str(uuid.uuid4())

    # Hash the UUID string using SHA-256
    hash_object = hashlib.sha256(uuid_str.encode())
    hex_dig = hash_object.hexdigest()
    return hex_dig[:hash_len]


def resize_to_nearest_multiple(image: Image.Image, divisor: int = 16) -> Image.Image:
    """
    Resize an image to the nearest size where both width and height are divisible by a given divisor.

    Args:
        image (Image.Image): The input image to resize.
        divisor (int): The number by which width and height should be divisible (default is 8).

    Returns:
        Image.Image: The resized image.
    """
    width, height = image.size

    # Calculate the nearest size divisible by the divisor
    new_width = (width + divisor - 1) // divisor * divisor
    new_height = (height + divisor - 1) // divisor * divisor

    # Resize the image
    resized_image = image.resize((new_width, new_height))
    return resized_image