import random

import gradio as gr
import numpy as np

# import spaces
import torch
from diffusers import (
    AutoencoderKL,
    AutoencoderTiny,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
)
from live_preview_helpers import (
    calculate_shift,
    flux_pipe_call_that_returns_an_iterable_of_images,
    retrieve_timesteps,
)
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

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


# @spaces.GPU(duration=75)
def infer(
    prompt,
    seed=42,
    randomize_seed=False,
    width=1024,
    height=1024,
    guidance_scale=3.5,
    num_inference_steps=28,
    progress=gr.Progress(track_tqdm=True),
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)

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


letter = 'B'
prompt = f"Create a stylized '{letter}'. The letter '{letter}' should appear in a magical, mystical style inspired by the Harry Potter universe. The design should incorporate subtle glowing effects, mystical symbols, and elegant curves, capturing the essence of fantasy and magic. Use dark and metallic tones, such as deep blacks, silvers, and hints of gold, to enhance the enchanting and mysterious appearance. Ensure the letter '{letter}' has intricate detailing, with an aged, slightly weathered texture that evokes an ancient, magical atmosphere. Letter '{letter}' should be on a clean white background. The design should be balanced and easily recognizable, with each letter in a consistent size and style."
seed = 127

img = infer(prompt=prompt, seed=seed)
img.save(f'{letter}_f.png')
