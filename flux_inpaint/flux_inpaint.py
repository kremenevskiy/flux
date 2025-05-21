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
    height: int = 1024,
    width: int = 1024,
    guidance_scale: float = 3.5,
    controlnet_conditioning_scale: float = 0.4,
    num_inference_steps: int = 28,
    true_guidance_scale: float = 1.0,
) -> None:
    if seed == 0:
        seed = random.randint(0, 1000)
        print('seed: ', seed)
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    generator = torch.Generator(device='cuda').manual_seed(seed)
    pipe = model_manager.get_model(model_name='flux_inpaint')

    params = {
        'prompt': prompt,
        'height': height,
        'width': width,
        'control_image': image,
        'control_mask': mask,
        'num_inference_steps': num_inference_steps,
        'generator': generator,
    }
    print('inpaint params')
    print(params)

    result = pipe(
        prompt=prompt,
        height=height,
        width=width,
        control_image=image,
        control_mask=mask,
        num_inference_steps=num_inference_steps,
        generator=generator,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        guidance_scale=guidance_scale,
        negative_prompt='',
        true_guidance_scale=true_guidance_scale,  # default: 3.5 for alpha and 1.0 for beta
    ).images[0]

    result.save(save_path)
    torch.cuda.empty_cache()
