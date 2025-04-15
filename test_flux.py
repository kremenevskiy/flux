import torch
from diffusers import DiffusionPipeline

model_id = 'black-forest-labs/FLUX.1-dev'
pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to('cuda')


lora_path = 'lora_models/lora_pic_e.safetensors'
pipeline.load_lora_weights(lora_path, adapter_name='lora_adapter')
style_lora_path = 'lora_models/lora_style.safetensors'
pipeline.load_lora_weights(style_lora_path, adapter_name='style')
pipeline.set_adapters(['style', 'lora_adapter'], adapter_weights=[0.8, 1.0])


p = 'pice style, blue tones, low importance of icon, slot icon, icon, small, Minimalistic blue mana potion vial emitting a soft, calming glow, gently resting on a serene alchemical table.'
g = torch.Generator(device='cpu').manual_seed(117)
img = pipeline(
    prompt=p,
    height=1000,
    width=1200,
    num_inference_steps=20,
    guidance_scale=3.5,
    generator=g,
).images[0]
img.save('test_flux_2.png')
