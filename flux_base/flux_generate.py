import random

import numpy as np

# import spaces
import torch
from diffusers import (
    AutoencoderKL,
    AutoencoderTiny,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
)
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

import model_manager
from flux_base.live_preview_helpers import (
    calculate_shift,
    flux_pipe_call_that_returns_an_iterable_of_images,
    retrieve_timesteps,
)


def get_model_pipe():
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

    MAX_SEED = np.iinfo(np.int32).max
    MAX_IMAGE_SIZE = 2048

    pipe.flux_pipe_call_that_returns_an_iterable_of_images = (
        flux_pipe_call_that_returns_an_iterable_of_images.__get__(pipe)
    )
    return pipe, good_vae


def get_model_pipe_with_lora():
    model_id = 'black-forest-labs/FLUX.1-dev'
    pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to('cuda')
    return pipeline


# @spaces.GPU(duration=75)
def infer(
    prompt: str,
    model_manager: model_manager.ModelManager,
    seed: int = 42,
    randomize_seed: bool = True,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 28,
) -> Image.Image:
    if randomize_seed and seed == 0:
        seed = random.randint(0, 10000)
    generator = torch.Generator().manual_seed(seed)

    # Get the model from the model manager
    if model_manager is None:
        raise ValueError('Model manager is required')
    pipe, good_vae = model_manager.get_model('flux_generate')
    meta = {
        'seed': seed,
        'guidance_scale': guidance_scale,
        'num_inference_steps': num_inference_steps,
    }
    print('params: ', meta)

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
    if tier == 'pic_1':
        return 'lora_models/lora_pic_a.safetensors'
    elif tier == 'pic_2':
        return 'lora_models/lora_pic_b.safetensors'
    elif tier == 'pic_3':
        return 'lora_models/lora_pic_c.safetensors'
    elif tier == 'pic_4' or tier == 'pic_5':
        return 'lora_models/lora_pic_e.safetensors'
    raise ValueError(f'Unknown tier: {tier}')


def get_trigger_word(tier: str) -> str:
    trigger_words = {
        'pic_1': 'pica style, luxurious style, gold, high importance, warm tones',
        'pic_2': 'picb style, large size of object, warm tones, high importance of icon, slot icon, icon, Large',
        'pic_3': 'picс style, medium size of object, violet tones, medium importance of icon, slot icon, icon, medium',
        'pic_4': 'picd style, green tones, low importance of icon, slot icon, icon, small',
        'pic_5': 'pice style, blue tones, low importance of icon, slot icon, icon, small',
    }
    if tier not in trigger_words:
        raise ValueError(f'Unknown tier: {tier}')

    return trigger_words[tier]


def infer_with_tier(
    prompt: str,
    model_manager: model_manager.ModelManager,
    tier: str | None = None,
    seed: int = 42,
    randomize_seed: bool = True,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 28,
) -> Image.Image:
    if randomize_seed and seed == 0:
        seed = random.randint(0, 10000)
    generator = torch.Generator().manual_seed(seed)

    # Get the model from the model manager
    if model_manager is None:
        raise ValueError('Model manager is required')

    pipe = model_manager.get_model('flux_generate_with_lora')

    lora_path = get_lora_path(tier)
    print('loading loras')

    pipe.unload_lora_weights()
    pipe.load_lora_weights(lora_path, adapter_name='lora_adapter')
    style_lora_path = 'lora_models/lora_style.safetensors'
    pipe.load_lora_weights(style_lora_path, adapter_name='style')
    pipe.set_adapters(['style', 'lora_adapter'], adapter_weights=[1.0, 1.0])

    meta = {
        'seed': seed,
        'guidance_scale': guidance_scale,
        'num_inference_steps': num_inference_steps,
        'lora_path': lora_path,
    }
    print('params: ', meta)

    prompt = f'{get_trigger_word(tier)}, {prompt}'

    img = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=20,
        guidance_scale=3.5,
        generator=generator,
    ).images[0]

    torch.cuda.empty_cache()
    return img
