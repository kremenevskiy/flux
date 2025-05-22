import hashlib

import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    'black-forest-labs/FLUX.1-dev', torch_dtype=torch.bfloat16
).to('cuda')


seed = 117
prompt = 'lowletters, The image is made in the style of bright retro signage, reminiscent of theater or circus posters. It features four large letters A, J, K and Q, each in its own rich color and with a bright contrasting border. "A" in red, "J" in purple, "K" in blue, "Q" in yellow. Glowing light bulbs are located inside the letters, which gives them a playful and festive character. The letter "A" is located in the upper left corner. The letter "K" is located in the lower left corner. The letter "J" is located in the upper right corner. The letter "Q" is located in the lower right corner. White background.'
prompt = 'lowletters, Three-dimensional letters with smooth curves, made of polished metal, which reflects the glare and creates a modern, almost futuristic look. Inside or on the surface of each letter there are contrasting inserts of saturated shades, "A" purple, "J" orange, "K" burgundy, "Q" red, resembling translucent inserts of colored glass or precious stones. The overall impression combines high-tech and fantasy aesthetics, giving the letters a dynamic, energetic look with touches of refined luxury. The letter "A" is located in the upper left corner. The letter "K" is located in the lower left corner. The letter "J" is located in the upper right corner. The letter "Q" is located in the lower right corner. White background.'

prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
prompt_4_hash = prompt_hash[:4]

img_base_path = f'/root/test/no_lora_{prompt_4_hash}.png'
img_with_lora_path = f'/root/test/with_lora_{prompt_4_hash}.png'
generator = torch.Generator(device='cpu').manual_seed(seed)

img = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=30,
    guidance_scale=3.5,
    generator=generator,
).images[0]

img.save(img_base_path)


pipe.unload_lora_weights()
pipe.load_lora_weights(
    '/root/flux/lora_models/lora_letters.safetensors',
    adapter_name='letters',
)
pipe.set_adapters(['letters'], adapter_weights=[1.0])

generator = torch.Generator(device='cpu').manual_seed(seed)

img = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=30,
    guidance_scale=3.5,
    generator=generator,
).images[0]

img.save(img_with_lora_path)
