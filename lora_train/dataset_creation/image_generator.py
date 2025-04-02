from diffusers import DiffusionPipeline
import torch
import json
from pathlib import Path
import random
from tqdm import tqdm

class ImageGenerator:
    def __init__(self, prompts_config_path: dict, save_images_dir_path: str, lora_path: str | None = None, prompt_prefix: str | None = None):
        self.prompts_config_path = prompts_config_path
        self.lora_path = lora_path
        self.load_model()
        self.generated_config_path = 'generated_config.json'
        self.save_images_dir_path = save_images_dir_path
        self.prompt_prefix = prompt_prefix

    def load_model(self):
        model_id = 'black-forest-labs/FLUX.1-dev'
        self.pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to('cuda')
        if self.lora_path:
            print('Loading lora weights...')
            self.pipe.load_lora_weights(self.lora_path, adapter_name="high")
            self.pipe.set_adapters(["high"], adapter_weights=[0.9])
            

    def generate_image(self, prompt: str, save_path: str):
        if self.lora_path:
            prompt = f'{self.prompt_prefix}, {prompt}'
        weights = [0.8, 0.85, 0.9]
        guidance_scale = [3.0, 3.25, 3.5]
        inference_steps = [20, 25, 28, 30]
        self.pipe.set_adapters(["high"], adapter_weights=[random.choice(weights)])
        image = self.pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=random.choice(inference_steps),
            guidance_scale=random.choice(guidance_scale),
        ).images[0]
        image.save(save_path)
        

    
    def process_prompt(self, prompt_data: dict, save_image_path: str) -> str:
        image_prompt = prompt_data['prompt']
        self.generate_image(image_prompt, save_image_path)
        new_prompt_data = prompt_data.copy()
        new_prompt_data['save_image_path'] = save_image_path
        return new_prompt_data
    
    
    def generate_images(self):
        with open(self.prompts_config_path, 'r') as f:
            prompts_config: list[dict] = json.load(f)
        

        if Path(self.generated_config_path).exists():
            with open(self.generated_config_path, 'r') as f:
                generated_config = json.load(f)
        else:
            generated_config = []
        
        for idx, prompt_data in tqdm(enumerate(prompts_config)):
            if idx % 20 == 0:
                print(f'Processing prompt {idx}/{len(prompts_config)}')
            if self.lora_path:
                save_image_path = f'{self.save_images_dir_path}/{prompt_data["tier"]}/{prompt_data["theme"]}_{prompt_data["icon"]}_{idx}.png'
            else:
                save_image_path = f'{self.save_images_dir_path}/{prompt_data["tier"]}_nolora/{prompt_data["theme"]}_{prompt_data["icon"]}_{idx}.png'
            save_image_path = save_image_path.replace('-', '_').replace(' ', '_').lower()
            Path(save_image_path).parent.mkdir(parents=True, exist_ok=True)
            if Path(save_image_path).exists():
                continue
            new_prompt_data = self.process_prompt(prompt_data, save_image_path)
            generated_config.append(new_prompt_data)
        
        with open(self.generated_config_path, 'w') as f:
            json.dump(generated_config, f, indent=4)
        


if __name__ == '__main__':
    prompts_config_path = '/root/flux/lora_train/dataset_creation/data/prompts_low_mid_tier_1.json'
    save_images_dir_path = '/root/flux/lora_train/dataset_creation/data/generated_images'
    lora_path = '/root/flux/lora_train/dataset_creation/lora_models/lora_style.safetensors'
    

    image_generator = ImageGenerator(
        prompts_config_path=prompts_config_path,
        save_images_dir_path=save_images_dir_path,
        lora_path=lora_path,
        # prompt_prefix='bbartstylecomp'
        prompt_prefix='bbartstylecomp, fantasy cartoon style'
    )
    image_generator.generate_images()