import random

import torch
from diffusers import DiffusionPipeline
from PIL import Image

import model_manager


def get_model_pipe_with_lora() -> DiffusionPipeline:
    model_id = 'black-forest-labs/FLUX.1-dev'
    return DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to('cuda')


def get_lora_path(lora_name: str) -> str:
    return f'lora_models/{lora_name}.safetensors'


def get_trigger_word(lora_name: str) -> str:
    return 'lowletters'


def load_lora_weights(
    pipe: DiffusionPipeline,
    lora_name: str,
    lora_strength: float,
) -> str:
    lora_path = get_lora_path(lora_name)

    pipe.unload_lora_weights()
    pipe.load_lora_weights(lora_path, adapter_name=lora_name)
    pipe.set_adapters([lora_name], adapter_weights=[lora_strength])

    return get_trigger_word(lora_name)


def infer_with_lora(
    prompt: str,
    model_manager: model_manager.ModelManager,
    lora_name: str,
    seed: int = 117,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 28,
    lora_strength: float = 1.0,
) -> Image.Image:
    if seed == 0:
        seed = random.randint(0, 10000)
    generator = torch.Generator().manual_seed(seed)

    pipe = model_manager.get_model('flux_generate_with_lora')
    trigger_word = load_lora_weights(
        pipe,
        lora_name=lora_name,
        lora_strength=lora_strength,
    )
    print(f'lora_name: {lora_name}, lora_strength: {lora_strength}, trigger_word: {trigger_word}')

    prompt = f'{trigger_word}, {prompt}'
    print(f'result prompt: {prompt}')
    img = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    torch.cuda.empty_cache()
    return img
