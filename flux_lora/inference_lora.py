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
    pipeline = DiffusionPipeline.from_pretrained(
        'black-forest-labs/FLUX.1-dev', torch_dtype=torch.bfloat16
    )
    pipeline.enable_model_cpu_offload()
    lora_weights_path = 'data/trained_loras/frames_lora_1.safetensors'
    pipeline.load_lora_weights(lora_weights_path)
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
    pipeline = model_manager.get_model('flux_lora')
    meta = {
        'seed': seed,
        'guidance_scale': guidance_scale,
        'num_inference_steps': num_inference_steps,
    }
    print('params: ', meta)

    img = pipeline(
        prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=512,
        generator=torch.Generator('cpu').manual_seed(seed),
        # cross_attention_kwargs={"scale": 0.9}
    ).images[0]

    torch.cuda.empty_cache()

    return img
