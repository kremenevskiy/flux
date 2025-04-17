import json
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from PIL import Image

from symbols_priority.icons_priority import gpt_api, icons_prompt


def get_trigger_word(tier: str, include_style: bool = False) -> str:
    trigger_words = {
        'pic_1': 'pic_a, luxurious style, gold, high importance, ultra close up',
        'pic_2': 'pic_b, large size of object, warm tones, high importance of icon, Large',
        'pic_3': 'pic_c, medium size of object, violet tones, medium importance of icon, medium',
        'pic_4': 'pic_d, green tones, slot, icon, small',
        'pic_5': 'pice style, blue tones, slot, icon, small',
        'style': 'bbartstylecomp',
    }
    if tier not in trigger_words:
        raise ValueError(f'Unknown tier: {tier}')

    trigger_word = trigger_words[tier]
    if include_style:
        trigger_word = f'{trigger_words["style"]}, {trigger_word}'
    return trigger_word


def get_lora_path(tier: str | None = None) -> str:
    lora_paths = {
        'pic_1': 'lora_models/lora_pic_a.safetensors',
        'pic_2': 'lora_models/lora_pic_b.safetensors',
        'pic_3': 'lora_models/lora_pic_c.safetensors',
        'pic_4': 'lora_models/lora_pic_d.safetensors',
        'pic_5': 'lora_models/lora_pic_e.safetensors',
    }

    if tier not in lora_paths:
        raise ValueError(f'Unknown tier: {tier}')

    return lora_paths[tier]


def load_lora_weights(
    pipe: DiffusionPipeline,
    tier: str,
    character_lora_strength: float,
    style_lora_strength: float,
) -> str:
    style_lora_path = 'lora_models/lora_style.safetensors'
    character_lora_path = get_lora_path(tier)

    pipe.unload_lora_weights()
    pipe.load_lora_weights(character_lora_path, adapter_name='character_lora')
    pipe.load_lora_weights(style_lora_path, adapter_name='style')
    pipe.set_adapters(
        ['style', 'character_lora'], adapter_weights=[style_lora_strength, character_lora_strength]
    )

    load_style_lora = style_lora_strength > 0.0
    return get_trigger_word(tier, include_style=load_style_lora)


pipe = None


def inference_with_lora(
    pipe: DiffusionPipeline,
    tier: str,
    prompt: str,
    seed: int = 117,
    character_lora_strength: float = 1.0,
    style_lora_strength: float = 0.0,
) -> Image.Image:
    generator = torch.Generator(device='cpu').manual_seed(seed)
    trigger_word = load_lora_weights(
        pipe=pipe,
        tier=tier,
        character_lora_strength=character_lora_strength,
        style_lora_strength=style_lora_strength,
    )

    prompt = f'{trigger_word}, {prompt}'
    return pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=30,
        guidance_scale=3.5,
        generator=generator,
    ).images[0]


class IconsTest:
    def __init__(self, themes_list: list[str], model_name: str, experiment_name: str):
        self.experiment_name = experiment_name
        self.experiment_path = Path('symbols_priority/icons_tests/') / experiment_name
        self.meta_save_path = self.experiment_path / 'icons_meta.json'
        self.themes_list = themes_list
        self.model_name = model_name
        self._init_pipe()

    def create_icons_prompts(self, theme: str) -> list[str]:
        return gpt_api.GptApi().get_sorted_symbols(theme, self.model_name)

    def save_prompts(self, prompts: list[str], save_path: str) -> None:
        data = {
            'prompts': prompts,
            'model_name': self.model_name,
            'system_prompt': icons_prompt.ICONS_SYSTEM_PROMPT,
        }
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=4)

    def generate_icons(self, prompts: list[str], save_dirpath: str) -> None:
        for pic_idx, prompt in enumerate(prompts, start=1):
            seeds = [117]
            for seed in seeds:
                image = inference_with_lora(
                    pipe=self.pipe, tier=f'pic_{pic_idx}', prompt=prompt, seed=seed
                )
                save_path = save_dirpath / f'pic_{pic_idx}' / f'{seed}.png'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(save_path)

    def run_experiment(self) -> None:
        for theme in self.themes_list:
            prompts = self.create_icons_prompts(theme)
            theme_name = theme.lower().replace(' ', '_')
            theme_path = self.experiment_path / theme_name
            theme_path.mkdir(parents=True, exist_ok=True)
            save_meta_path = theme_path / 'icons_meta.json'
            self.save_prompts(prompts, save_meta_path)
            self.generate_icons(prompts, theme_path)

    def _init_pipe(self) -> None:
        self.pipe = DiffusionPipeline.from_pretrained(
            'black-forest-labs/FLUX.1-dev', torch_dtype=torch.bfloat16
        ).to('cuda')


def run_experiment(themes_list: list[str], experiment_name: str, model_name: str) -> None:
    icons_test = IconsTest(themes_list, model_name, experiment_name)
    icons_test.run_experiment()


def main() -> None:
    exp_name = 'base'
    themes_list = ['Anime', 'New Year']
    model_name = 'gpt-4o-mini'
    run_experiment(themes_list, exp_name, model_name)


if __name__ == '__main__':
    main()
