import random

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    AutoencoderTiny,
    DiffusionPipeline,
)
from PIL import Image

import model_manager
from flux_base.live_preview_helpers import flux_pipe_call_that_returns_an_iterable_of_images


def get_model_pipe() -> tuple[DiffusionPipeline, AutoencoderKL]:
    dtype = torch.float16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    taef1 = AutoencoderTiny.from_pretrained('madebyollin/taef1', torch_dtype=dtype).to(device)
    good_vae = AutoencoderKL.from_pretrained(
        'black-forest-labs/FLUX.1-dev', subfolder='vae', torch_dtype=dtype
    ).to(device)
    pipe = DiffusionPipeline.from_pretrained(
        'black-forest-labs/FLUX.1-dev', torch_dtype=dtype, vae=taef1
    ).to(device)
    torch.cuda.empty_cache()

    pipe.flux_pipe_call_that_returns_an_iterable_of_images = (
        flux_pipe_call_that_returns_an_iterable_of_images.__get__(pipe)
    )
    return pipe, good_vae


def get_model_pipe_with_lora() -> DiffusionPipeline:
    model_id = 'black-forest-labs/FLUX.1-dev'
    return DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to('cuda')


def infer(
    prompt: str,
    model_manager: model_manager.ModelManager,
    seed: int = 42,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 28,
) -> Image.Image:
    if seed == 0:
        seed = random.randint(0, 10000)
    generator = torch.Generator().manual_seed(seed)

    pipe, good_vae = model_manager.get_model('flux_generate')

    for img in pipe.flux_pipe_call_that_returns_an_iterable_of_images(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=generator,
        output_type='pil',
        good_vae=good_vae,
    ):
        img, seed = img, seed

    torch.cuda.empty_cache()
    return img


def get_lora_path(tier: str | None = None) -> str:
    lora_paths = {
        'pic_1': 'lora_models/lora_pic_a.safetensors',
        'pic_2': 'lora_models/lora_pic_b.safetensors',
        'pic_3': 'lora_models/lora_pic_c.safetensors',
        'pic_4': 'lora_models/lora_pic_d.safetensors',
        'pic_5': 'lora_models/lora_pic_e.safetensors',
    }

    if tier not in lora_paths:
        raise ValueError(f'Unknown tier: {tier}')

    return lora_paths[tier]


def get_trigger_word(tier: str, include_style: bool = False) -> str:
    trigger_words = {
        'pic_1': 'pic_a, luxurious style, gold, high importance, ultra close up',
        'pic_2': 'pic_b, large size of object, warm tones, high importance of icon, Large',
        'pic_3': 'pic_c, medium size of object, violet tones, medium importance of icon, medium',
        'pic_4': 'pic_d, green tones, slot, icon, small',
        'pic_5': 'pice style, blue tones, slot, icon, small',
        'style': 'bbartstylecomp',
    }
    if tier not in trigger_words:
        raise ValueError(f'Unknown tier: {tier}')

    trigger_word = trigger_words[tier]
    if include_style:
        trigger_word = f'{trigger_words["style"]}, {trigger_word}'
    return trigger_word


def load_lora_weights(
    pipe: DiffusionPipeline,
    tier: str,
    character_lora_strength: float,
    style_lora_strength: float,
) -> str:
    style_lora_path = 'lora_models/lora_style.safetensors'
    character_lora_path = get_lora_path(tier)

    pipe.unload_lora_weights()
    pipe.load_lora_weights(character_lora_path, adapter_name='character_lora')
    pipe.load_lora_weights(style_lora_path, adapter_name='style')
    pipe.set_adapters(
        ['style', 'character_lora'], adapter_weights=[style_lora_strength, character_lora_strength]
    )

    load_style_lora = style_lora_strength > 0.0
    return get_trigger_word(tier, include_style=load_style_lora)


def infer_with_tier(
    prompt: str,
    model_manager: model_manager.ModelManager,
    tier: str | None = None,
    seed: int = 117,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 28,
    character_lora_strength: float = 1.0,
    style_lora_strength: float = 0.0,
) -> Image.Image:
    if seed == 0:
        seed = random.randint(0, 10000)
    generator = torch.Generator().manual_seed(seed)

    pipe = model_manager.get_model('flux_generate_with_lora')
    trigger_word = load_lora_weights(
        pipe,
        tier=tier,
        character_lora_strength=character_lora_strength,
        style_lora_strength=style_lora_strength,
    )
    print(
        f'tier: {tier}, trigger_word: {trigger_word}, character_lora_strength: {character_lora_strength}, style_lora_strength: {style_lora_strength}'
    )

    prompt = f'{trigger_word}, {prompt}'
    img = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=30,  # num_inference_steps
        guidance_scale=3.5,  # guidance_scale
        generator=generator,
    ).images[0]

    torch.cuda.empty_cache()
    return img
