import json
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from PIL import Image, ImageDraw, ImageFont

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
    print(prompt)
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

    def save_prompts(self, prompts: list[str], icons_list: list[str], save_path: str) -> None:
        data = {
            'icons_list': icons_list,
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
                save_path = save_dirpath / f'pic_{pic_idx}' / f'{seed}.png'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                if save_path.exists():
                    print(f'skipping {save_path}, file already exists')
                    continue
                image = inference_with_lora(
                    pipe=self.pipe, tier=f'pic_{pic_idx}', prompt=prompt, seed=seed
                )

                image.save(save_path)

    def run_experiment(self) -> None:
        for theme in self.themes_list:
            print(f'running experiment for {theme}')
            theme_name = theme.lower().replace(' ', '_')
            theme_path = self.experiment_path / theme_name
            save_meta_path = theme_path / 'icons_meta.json'
            if not save_meta_path.exists():
                prompts_config = self.create_icons_prompts(theme)
                prompts = list(prompts_config['icon_prompts'].values())
                theme_path.mkdir(parents=True, exist_ok=True)
                self.save_prompts(
                    prompts=prompts,
                    icons_list=prompts_config['slot_icons'],
                    save_path=save_meta_path,
                )
            else:
                with open(save_meta_path) as f:
                    prompts_config = json.load(f)

                prompts = prompts_config['prompts']

            self.generate_icons(prompts, theme_path)

    def make_summary(self) -> None:
        summary_file_path = self.experiment_path / 'summary.png'
        img_size = 256
        padding = 10
        text_height = 30  # Height allocated for theme text
        cols = 5  # pic_1 to pic_5
        rows = len(self.themes_list)

        total_width = cols * img_size + (cols + 1) * padding
        # Adjust height for images, padding, and text for each row
        total_height = rows * (img_size + text_height) + (rows + 1) * padding

        # Create a white background canvas
        summary_image = Image.new('RGB', (total_width, total_height), color='white')
        draw = ImageDraw.Draw(summary_image)

        # Try to load a default font, fall back to a basic one if needed
        try:
            font = ImageFont.truetype('arial.ttf', 20)  # Try loading Arial font, size 20
        except IOError:
            print('Arial font not found. Using default PIL font.')
            font = ImageFont.load_default()

        placeholder_image = Image.new(
            'RGB', (img_size, img_size), color=(200, 200, 200)
        )  # Gray placeholder

        y_offset = padding
        for theme_idx, theme in enumerate(self.themes_list):
            theme_name = theme
            theme_path_name = theme.lower().replace(' ', '_')
            theme_path = self.experiment_path / theme_path_name
            x_offset = padding

            # Paste images for the current row
            for pic_idx in range(1, 6):
                image_filename = '117.png'  # Using the first seed's image
                image_path = theme_path / f'pic_{pic_idx}' / image_filename

                if image_path.exists():
                    try:
                        img = Image.open(image_path).convert('RGB')
                        img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
                        summary_image.paste(img, (x_offset, y_offset))
                    except Exception as e:
                        print(f'Error processing image {image_path}: {e}')
                        summary_image.paste(placeholder_image, (x_offset, y_offset))
                else:
                    # Paste placeholder if image not found
                    summary_image.paste(placeholder_image, (x_offset, y_offset))

                x_offset += img_size + padding

            # Draw the theme text below the row of images
            text_y_position = y_offset + img_size + (padding // 2)  # Position text below images
            text_x_position = padding  # Align text to the left padding
            draw.text((text_x_position, text_y_position), theme_name, fill='black', font=font)

            # Update y_offset for the next row (including image height, text height, and padding)
            y_offset += img_size + text_height + padding

        try:
            summary_image.save(summary_file_path)
            print(f'Summary PNG generated at: {summary_file_path}')
        except IOError as e:
            print(f'Error saving summary PNG: {e}')

    def run(self) -> None:
        self.run_experiment()
        self.make_summary()

    def _init_pipe(self) -> None:
        self.pipe = DiffusionPipeline.from_pretrained(
            'black-forest-labs/FLUX.1-dev', torch_dtype=torch.bfloat16
        ).to('cuda')


def run_experiment(themes_list: list[str], experiment_name: str, model_name: str) -> None:
    icons_test = IconsTest(themes_list, model_name, experiment_name)
    icons_test.run()


def main() -> None:
    exp_name = 'gpt_4.1_with_alive_first'
    themes_list = [
        # 'Brawl Stars',
        # 'Neon Anime Adventure',
        # 'Elegant Origami Creations',
        # 'Adult Night Passion',
        # 'Mystic Mountain Peaks',
        # 'Groovy Dance Party',
        # 'Melodic Note Symphony',
        # 'Crispy Potato Pancakes',
        # 'Global Food Feast',
        # 'Cute Kitten Paradise',
        # 'Happy Puppy Park',
        # 'Blooming Flower Garden',
        'Haunted Ghost Mansion',
        'Belarusian Mythic Legends',
        # 'Champion Boxing Ring',
        'Harry Potter',
        # 'Hallo Kitty',
        'Halloween Night',
        # 'Micky Mouse',
        # 'Paw Patrol',
        # 'SpongeBob SquarePants',
        # 'The Simpsons',
        'The Witches',
        # 'The Wizard of Oz',
        'The Lord of the Rings',
        'The Hobbit',
        'The Chronicles of Narnia',
        'Pirates of the Caribbean',
        'Game of Thrones',
        'BELARUS HATES TRUMP',
        'Stock Market down because of Trump',
        'Ancient Egyptian Treasures',
        'Wild West Outlaws',
        'Deep Sea Exploration',
        'Tropical Paradise Resort',
        'Futuristic Cyberpunk City',
        'Medieval Fantasy Kingdom',
        'Viking Warriors Voyage',
        'Jurassic Dinosaur World',
        'Magical Fairy Forest',
        'Cosmic Space Adventure',
        'Golden Chinese Dynasty',
        'Post-Apocalyptic Wasteland',
        'Steampunk Inventors',
        'Mythical Greek Gods',
        'Aztec Temple Mysteries',
        'Enchanted Candy Land',
        'Retro Arcade Games',
        'Spicy Mexican Fiesta',
        'Luxury Casino Lifestyle',
        'Robot Uprising Revolution',
        'Superhero Team Battle',
        'Classic Horror Monsters',
        'Samurai Honor Code',
        'Underwater Mermaid Kingdom',
        'Olympic Sports Champions',
        'Exotic Jungle Safari',
        'Frozen Arctic Expedition',
        'Magic Circus Performers',
        'Vintage Hollywood Stars',
        'Swashbuckling Pirate Adventure',
    ]
    model_name = 'gpt-4.1'
    run_experiment(themes_list, exp_name, model_name)


if __name__ == '__main__':
    main()
