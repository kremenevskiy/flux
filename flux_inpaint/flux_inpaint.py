import random

import torch
from diffusers.utils import check_min_version
from PIL import Image

import model_manager
from flux_inpaint.controlnet_flux import FluxControlNetModel
from flux_inpaint.pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
from flux_inpaint.transformer_flux import FluxTransformer2DModel

check_min_version('0.30.2')


def get_model_pipe() -> FluxControlNetInpaintingPipeline:
    controlnet = FluxControlNetModel.from_pretrained(
        'alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha', torch_dtype=torch.bfloat16
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        'black-forest-labs/FLUX.1-dev', subfolder='transformer', torch_dtype=torch.bfloat16
    )
    pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        'black-forest-labs/FLUX.1-dev',
        controlnet=controlnet,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    ).to('cuda')
    pipe.transformer.to(torch.bfloat16)
    pipe.controlnet.to(torch.bfloat16)
    return pipe


def create_inpaint(
    prompt: str,
    image_path: str,
    mask_path: str,
    save_path: str,
    model_manager: model_manager.ModelManager,
    seed: int = 24,
) -> None:
    if seed == 0:
        seed = random.randint(0, 1000)
        print('seed: ', seed)
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    generator = torch.Generator(device='cuda').manual_seed(seed)
    pipe = model_manager.get_model(model_name='flux_inpaint')
    result = pipe(
        prompt=prompt,
        height=image.size[1],
        width=image.size[0],
        control_image=image,
        control_mask=mask,
        num_inference_steps=28,
        generator=generator,
        controlnet_conditioning_scale=0.9,
        guidance_scale=3.5,
        negative_prompt='',
        true_guidance_scale=1.0,  # default: 3.5 for alpha and 1.0 for beta
    ).images[0]

    result.save(save_path)
