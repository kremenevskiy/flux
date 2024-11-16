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


# @spaces.GPU(duration=75)
def infer(
    prompt: str,
    model_manager: model_manager.ModelManager,
    seed: int = 42,
    randomize_seed: bool = False,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 28,
) -> Image.Image:
    if randomize_seed:
        seed = random.randint(0, 10000)
    generator = torch.Generator().manual_seed(seed)

    # Get the model from the model manager
    if model_manager is None:
        raise ValueError('Model manager is required')
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
    return img
