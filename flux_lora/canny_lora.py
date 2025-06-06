import random

import torch
from controlnet_aux import CannyDetector
from diffusers import FluxControlPipeline
from diffusers.utils import load_image
from PIL import Image

import model_manager


def get_lora_path(lora_name: str) -> str:
    lora_paths = {
        'reelslora_old': 'lora_models/frames_lora_1.safetensors',
        'reels_lora_new': 'reelslorafluxv13.safetensors',
    }

    if lora_name not in lora_paths:
        raise ValueError(f'Unknown lora: {lora_name}')

    return lora_paths[lora_name]


def get_trigger_word(lora_name: str) -> str:
    trigger_words = {
        'reelslora_old': 'reelslora',
        'reels_lora_new': 'reelslora',
    }
    if lora_name not in trigger_words:
        raise ValueError(f'Unknown lora: {lora_name}')

    return trigger_words[lora_name]


def load_lora_weights(
    pipe: FluxControlPipeline,
    lora_name: str,
    canny_lora_strength: float = 0.8,
    frames_lora_strength: float = 1.0,
) -> str:
    frames_lora_path = get_lora_path(lora_name)

    pipe.unload_lora_weights()
    pipe.load_lora_weights(frames_lora_path, adapter_name='frames')
    pipe.load_lora_weights('black-forest-labs/FLUX.1-Canny-dev-lora', adapter_name='canny')

    pipe.set_adapters(
        ['canny', 'frames'], adapter_weights=[canny_lora_strength, frames_lora_strength]
    )

    return get_trigger_word(lora_name)


def get_model_pipe() -> FluxControlPipeline:
    return FluxControlPipeline.from_pretrained(
        'black-forest-labs/FLUX.1-dev', torch_dtype=torch.bfloat16
    ).to('cuda')


def inference_frame_canny_with_lora(
    prompt: str,
    control_image_path: str,
    model_manager: model_manager.ModelManager,
    lora_name: str,
    canny_lora_strength: float,
    frames_lora_strength: float,
    seed: int = 117,
    width: int | None = None,
    height: int | None = None,
    guidance_scale: float = 30.0,
    num_inference_steps: int = 30,
) -> Image.Image:
    control_image = load_image(control_image_path)

    width, height = control_image.size if (width is None and height is None) else (width, height)
    processor = CannyDetector()
    control_image = processor(
        control_image,
        low_threshold=50,
        high_threshold=200,
        detect_resolution=1024,
        image_resolution=min(width, height),
    )

    pipe = model_manager.get_model('flux_generate_canny_with_lora')
    trigger_word = load_lora_weights(
        pipe,
        lora_name=lora_name,
        canny_lora_strength=canny_lora_strength,
        frames_lora_strength=frames_lora_strength,
    )

    if seed == 0:
        seed = random.randint(0, 10000)
    generator = torch.Generator().manual_seed(seed)

    prompt = f'{trigger_word}, {prompt}'
    img = pipe(
        prompt=prompt,
        control_image=control_image,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    torch.cuda.empty_cache()
    return img
