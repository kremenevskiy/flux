import json
from pathlib import Path

import gradio as gr
import numpy as np
import requests
from gradio_imageslider import ImageSlider  # Import ImageSlider

# import spaces
from image_datasets.canny_dataset import c_crop, canny_processor
from PIL import Image
from src.flux.sampling import denoise_controlnet, get_noise, get_schedule, prepare, unpack

import utils


def preprocess_image(image, target_width, target_height, crop=True):
    if crop:
        image = c_crop(image)  # Crop the image to square
        original_width, original_height = image.size

        # Resize to match the target size without stretching
        scale = max(target_width / original_width, target_height / original_height)
        resized_width = int(scale * original_width)
        resized_height = int(scale * original_height)

        image = image.resize((resized_width, resized_height), Image.LANCZOS)

        # Center crop to match the target dimensions
        left = (resized_width - target_width) // 2
        top = (resized_height - target_height) // 2
        image = image.crop((left, top, left + target_width, top + target_height))
    else:
        image = image.resize((target_width, target_height), Image.LANCZOS)

    return image


def preprocess_canny_image(image, target_width, target_height, crop=True):
    image = preprocess_image(image, target_width, target_height, crop=crop)
    image = canny_processor(image)
    return image


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


FLUX_SERVER_ENDPOINT = 'http://38.29.145.18:40919/flux-canny-image'
FLUX_TIMEOUT = 300


def flux_get_image_from_canny(
    prompt: str,
    image_path: str,
    save_path: str,
    seed: int = 24,
    num_steps: int = 50,
    guidance: float = 4.0,
    canny_guidance: float = 0.7,
) -> str:
    with Path(image_path).open('rb') as image_file:
        files = {
            'image': (Path(image_path).name, image_file, 'image/jpeg'),  # Main image file
        }
        data = {
            'prompt': prompt,
            'seed': seed,
            'num_steps': num_steps,
            'guidance': guidance,
            'canny_guidance': canny_guidance,
        }

        response = requests.post(
            FLUX_SERVER_ENDPOINT,
            files=files,
            data=data,
            timeout=FLUX_TIMEOUT,
        )

        if response.status_code == 200:
            with Path(save_path).open('wb') as output_file:
                output_file.write(response.content)

        # unique_dir = utils.get_hash_from_uuid(hash_len=7)
        # gen_dir = Path(res_dir) / unique_dir
        # gen_dir.mkdir(parents=True, exist_ok=True)

        # control_image_save_path = gen_dir / 'contol_image.png'
        # result_image_save_path = gen_dir / 'result.png'

        # control_image = Image.open(image_path)
        # control_image.save(control_image_save_path)
        # output_img = Image.open(save_path)
        # output_img.save(result_image_save_path)

        # meta_json_path = gen_dir / 'meta.json'
        # with open(meta_json_path, 'w') as json_file:
        #     json.dump(data, json_file, indent=4)

        return save_path


res_dir = 'gradio_dir'


# @spaces.GPU(duration=120)
def generate_image(
    prompt,
    control_image,
    num_steps=50,
    guidance=4,
    width=512,
    height=512,
    seed=42,
    random_seed=False,
    controlnet_gs=0.7,
):
    if random_seed:
        seed = np.random.randint(0, 10000)

    unique_dir = utils.get_hash_from_uuid(hash_len=7)
    gen_dir = Path(res_dir) / unique_dir
    gen_dir.mkdir(parents=True, exist_ok=True)
    control_image_save_path = gen_dir / 'contol_image.png'

    if width == 512 and height == 512:
        width, height = control_image.size
    else:
        control_image = control_image.resize((width, height))
        control_image.save(control_image_save_path)

    control_image = resize_to_nearest_multiple(image=control_image, divisor=16)

    canny_processed_tmp = preprocess_canny_image(control_image, width, height, crop=False)

    control_image_save_path = gen_dir / 'contol_image.png'
    canny_image_save_path = gen_dir / 'canny.png'
    result_image_save_path = gen_dir / 'result.png'

    control_image.save(control_image_save_path)
    canny_processed_tmp.save(canny_image_save_path)

    flux_get_image_from_canny(
        prompt=prompt,
        image_path=control_image_save_path,
        save_path=result_image_save_path,
        seed=seed,
        num_steps=num_steps,
        guidance=guidance,
        canny_guidance=controlnet_gs,
    )
    output_img = Image.open(result_image_save_path)

    meta = {
        'prompt': prompt,
        'control_image_path': str(control_image_save_path),
        'canny_path': str(canny_image_save_path),
        'result_path': str(result_image_save_path),
        'num_steps': num_steps,
        'guidance': guidance,
        'controlnet_gs': controlnet_gs,
        'width': width,
        'height': height,
        'seed': seed,
    }

    meta_json_path = gen_dir / 'meta.json'
    with open(meta_json_path, 'w') as json_file:
        json.dump(meta, json_file, indent=4)

    return [canny_processed_tmp, output_img]  # Return both images for slider


interface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label='Prompt'),
        gr.Image(type='pil', label='Control Image'),
        gr.Slider(step=1, minimum=1, maximum=64, value=28, label='Num Steps'),
        gr.Slider(minimum=0.1, maximum=30, value=4, label='Guidance'),
        gr.Slider(minimum=128, maximum=1024, step=128, value=512, label='Width'),
        gr.Slider(minimum=128, maximum=1024, step=128, value=512, label='Height'),
        gr.Slider(minimum=0, maximum=9999999, step=1, value=42, label='Seed'),
        #       gr.Number(value=42, label="Seed"),
        gr.Checkbox(label='Random Seed'),
        gr.Slider(minimum=0, maximum=2, value=0.7, label='controlnet_gs'),
    ],
    outputs=ImageSlider(label='Before / After'),  # Use ImageSlider as the output
    title='FLUX.1 Controlnet Canny',
    description='Generate images using ControlNet and a text prompt.\n[[non-commercial license, Flux.1 Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)]',
)

if __name__ == '__main__':
    interface.launch(server_port=11235, server_name='0.0.0.0')
