import os
import torch
import gradio as gr
import numpy as np
from PIL import Image
from einops import rearrange
import requests
from huggingface_hub import login
from gradio_imageslider import ImageSlider  # Import ImageSlider

from image_datasets.canny_dataset import canny_processor, c_crop
from src.flux.sampling import denoise_controlnet, get_noise, get_schedule, prepare, unpack
from src.flux.util import load_ae, load_clip, load_t5, load_flow_model, load_controlnet, load_safetensors

# Download and load the ControlNet model
model_url = "https://huggingface.co/XLabs-AI/flux-controlnet-canny-v3/resolve/main/flux-canny-controlnet-v3.safetensors?download=true"
model_path = "./flux-canny-controlnet-v3.safetensors"
if not os.path.exists(model_path):
    response = requests.get(model_url)
    with open(model_path, 'wb') as f:
        f.write(response.content)

# Source: https://github.com/XLabs-AI/x-flux.git
name = "flux-dev"
device = torch.device("cuda")
offload = False
is_schnell = name == "flux-schnell"

def preprocess_image(image, target_width, target_height, crop=True):
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

def generate_image(prompt, control_image, num_steps=50, guidance=4, width=512, height=512, seed=42, random_seed=False):
    if random_seed:
        seed = np.random.randint(0, 10000)
    
    if not os.path.isdir("./controlnet_results/"):
        os.makedirs("./controlnet_results/")

    torch_device = torch.device("cuda")
    
    torch.cuda.empty_cache()  # Clear GPU cache
    
    model = load_flow_model(name, device=torch_device)
    t5 = load_t5(torch_device, max_length=256 if is_schnell else 512)
    clip = load_clip(torch_device)
    ae = load_ae(name, device=torch_device)
    controlnet = load_controlnet(name, torch_device).to(torch_device).to(torch.bfloat16)

    checkpoint = load_safetensors(model_path)
    controlnet.load_state_dict(checkpoint, strict=False)

    width = 16 * width // 16
    height = 16 * height // 16
    timesteps = get_schedule(num_steps, (width // 8) * (height // 8) // (16 * 16), shift=(not is_schnell))
    
    processed_input = preprocess_image(control_image, width, height)
    canny_processed = preprocess_canny_image(control_image, width, height)
    controlnet_cond = torch.from_numpy((np.array(canny_processed) / 127.5) - 1)
    controlnet_cond = controlnet_cond.permute(2, 0, 1).unsqueeze(0).to(torch.bfloat16).to(torch_device)

    torch.manual_seed(seed)
    with torch.no_grad():
        x = get_noise(1, height, width, device=torch_device, dtype=torch.bfloat16, seed=seed)
        inp_cond = prepare(t5=t5, clip=clip, img=x, prompt=prompt)

        x = denoise_controlnet(model, **inp_cond, controlnet=controlnet, timesteps=timesteps, guidance=guidance, controlnet_cond=controlnet_cond)
        
        x = unpack(x.float(), height, width)
        x = ae.decode(x)

    x1 = x.clamp(-1, 1)
    x1 = rearrange(x1[-1], "c h w -> h w c")
    output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())
    
    return [processed_input, output_img]  # Return both images for slider

interface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Image(type="pil", label="Control Image"),
        gr.Slider(step=1, minimum=1, maximum=64, value=28, label="Num Steps"),
        gr.Slider(minimum=0.1, maximum=10, value=4, label="Guidance"),
        gr.Slider(minimum=128, maximum=1024, step=128, value=512, label="Width"),
        gr.Slider(minimum=128, maximum=1024, step=128, value=512, label="Height"),
        gr.Slider(minimum=0, maximum=9999999, step=1, value=42, label="Seed"),
#       gr.Number(value=42, label="Seed"),
        gr.Checkbox(label="Random Seed")
    ],
    outputs=ImageSlider(label="Before / After"),  # Use ImageSlider as the output
    title="FLUX.1 Controlnet Canny",
    description="Generate images using ControlNet and a text prompt.\n[[non-commercial license, Flux.1 Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)]"
)

if __name__ == "__main__":
    interface.launch()