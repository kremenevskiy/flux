import os

import numpy as np
import requests
import torch
from einops import rearrange
from PIL import Image

import model_manager
import utils_service
from flux_canny.image_datasets.canny_dataset import c_crop, canny_processor
from flux_canny.src.flux.sampling import (
    denoise_controlnet,
    get_noise,
    get_schedule,
    prepare,
    unpack,
)
from flux_canny.src.flux.util import (
    load_ae,
    load_clip,
    load_controlnet,
    load_flow_model,
    load_safetensors,
    load_t5,
)


def get_model_pipe() -> tuple:
    # Download and load the ControlNet model
    model_url = 'https://huggingface.co/XLabs-AI/flux-controlnet-canny-v3/resolve/main/flux-canny-controlnet-v3.safetensors?download=true'
    model_path = './flux-canny-controlnet-v3.safetensors'
    if not os.path.exists(model_path):
        response = requests.get(model_url)
        with open(model_path, 'wb') as f:
            f.write(response.content)

    # Source: https://github.com/XLabs-AI/x-flux.git
    name = 'flux-dev'
    device = torch.device('cuda')
    offload = False
    is_schnell = name == 'flux-schnell'

    torch.cuda.empty_cache()  # Clear GPU cache

    torch_device = torch.device('cuda')

    model = load_flow_model(name, device=torch_device)
    t5 = load_t5(torch_device, max_length=256 if is_schnell else 512)
    clip = load_clip(torch_device)
    ae = load_ae(name, device=torch_device)
    controlnet = load_controlnet(name, torch_device).to(torch_device).to(torch.bfloat16)

    checkpoint = load_safetensors(model_path)
    controlnet.load_state_dict(checkpoint, strict=False)

    return model, t5, clip, ae, controlnet, is_schnell


def create_canny(
    prompt: str,
    control_image_path: str,
    save_path: str,
    model_manager: model_manager.ModelManager,
    num_steps: int = 50,
    guidance: float = 4,
    seed: int = 24,
    width: int | None = None,
    height: int | None = None,
    canny_guidance: float = 0.7,
) -> None:
    meta = {
        'num_steps': num_steps,
        'guidance': guidance,
        'controlnet_gs': canny_guidance,
    }
    print('params: ', meta)

    control_image = Image.open(control_image_path)
    control_image = utils_service.resize_to_nearest_multiple(image=control_image)
    control_image.save('default.png')

    width, height = control_image.size if (width is None and height is None) else (width, height)
    torch_device = torch.device('cuda')

    if not os.path.isdir('./controlnet_results/'):
        os.makedirs('./controlnet_results/')

    model, t5, clip, ae, controlnet, is_schnell = model_manager.get_model(model_name='flux_canny')
    width = 16 * width // 16
    height = 16 * height // 16
    timesteps = get_schedule(
        num_steps, (width // 8) * (height // 8) // (16 * 16), shift=(not is_schnell)
    )

    print('timesteps: ', timesteps)

    canny_processed = preprocess_canny_image(control_image, width, height, crop=False)
    canny_processed.save('canny.png')
    controlnet_cond = torch.from_numpy((np.array(canny_processed) / 127.5) - 1)
    controlnet_cond = (
        controlnet_cond.permute(2, 0, 1).unsqueeze(0).to(torch.bfloat16).to(torch_device)
    )

    torch.manual_seed(seed)
    with torch.no_grad():
        x = get_noise(1, height, width, device=torch_device, dtype=torch.bfloat16, seed=seed)
        inp_cond = prepare(t5=t5, clip=clip, img=x, prompt=prompt)

        x = denoise_controlnet(
            model,
            **inp_cond,
            controlnet=controlnet,
            timesteps=timesteps,
            guidance=guidance,
            controlnet_cond=controlnet_cond,
            controlnet_gs=canny_guidance,
        )

        x = unpack(x.float(), height, width)
        x = ae.decode(x)

    x1 = x.clamp(-1, 1)
    x1 = rearrange(x1[-1], 'c h w -> h w c')
    output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())

    output_img.save(save_path)


def preprocess_image(image, target_width, target_height, crop=False):
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


# prompt = 'Design a high-resolution background frame for a slot machine game, inspired by the magical, humorous world of Shrek. The frame should feature medieval architectural elements intertwined with swamp foliage, such as vines and lily pads. Include charming details like Puss in Boots peeking from a corner or Gingerbread Man characters integrated into the frame’s design. Use a bright, cartoonish art style with bold colors and soft shading to emulate the animation style of the Shrek films'
# control_image_path = '/root/flux-canny/default.png'
# save_path = 'res.png'
# create_canny(prompt=prompt, control_image_path=control_image_path, save_path=save_path)
