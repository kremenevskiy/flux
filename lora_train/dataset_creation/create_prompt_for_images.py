from loguru import logger
from lora_train.dataset_creation import gpt_api, gpt_prompts
import json
from pathlib import Path
import asyncio

class DatasetCreator:
    def __init__(self):
        self.gpt_api = gpt_api.GptApi()

    async def generate_style_icons(self, previous_themes: list[dict]) -> dict:
            style_data = await self.gpt_api.ask_gpt(
                system_prompt=gpt_prompts.build_theme_generator_prompt(previous_themes),
                user_prompt="Generate a new unique theme and icons.",
            )    
            return style_data

    async def generate_and_save_style_icons(self, num_styles: int,  previous_themes_path: str, output_path: str) -> None:
        with open(previous_themes_path, 'r') as f:
            generated_themes: list[dict] = json.load(f)

        for _ in range(num_styles):
             new_theme = await self.generate_style_icons(previous_themes=generated_themes)
             generated_themes.append(new_theme)
        
        with open(output_path, 'w') as f:
            json.dump(generated_themes, f, indent=4)
        logger.info(f"Saved {num_styles} style icons to {output_path}")



class PromptCreator:
    def __init__(self):
        self.gpt_api = gpt_api.GptApi()
        self.style_tiers = ['top-tier', 'high-tier', 'mid-tier', 'low-mid-tier', 'low-tier']


    async def generate_prompts_for_icon(self, theme: str, icon: str) -> list[dict]:
        prompts = []
        for tier in self.style_tiers:
            prompt = await self.gpt_api.ask_gpt(
                system_prompt=gpt_prompts.build_system_prompt(tier, theme, icon),
                user_prompt=f"Theme: {theme}\nIcon: {icon}\nStylistic tier: {tier}",
                return_json=False

            )
            prompts.append({
                "theme": theme,
                "icon": icon,
                "tier": tier,
                "prompt": prompt
            })
        return prompts
    
    def _is_icon_already_processed(self, icon: str, theme: str, images_promts: list[dict]) -> bool:
        for prompt in images_promts:
            if prompt["icon"] == icon and prompt["theme"] == theme:
                return True
        return False

    async def generate_and_save_style_icons(self, themes_path: str, output_prompt_path: str, num_icons_to_generate: int = 100) -> None:
        with open(themes_path, 'r') as f:
            generated_themes: list[dict] = json.load(f)

        if Path(output_prompt_path).exists():
            with open(output_prompt_path, 'r') as f:
                images_promts: list[dict] = json.load(f)
        else:
            images_promts = []
        i = 0
        should_stop = False
        for new_theme in generated_themes:
            theme = new_theme["theme"]
            for icon in new_theme["icons"]:
                i += 1
                if i > num_icons_to_generate:
                    should_stop = True
                    break

                if self._is_icon_already_processed(icon, theme, images_promts):
                    print(f"Icon {icon} for theme {theme} already processed")
                    continue
                icon_prompts = await self.generate_prompts_for_icon(theme, icon)

                images_promts.extend(icon_prompts)

                
            if should_stop:
                break

        with open(output_prompt_path, 'w') as f:
            json.dump(images_promts, f, indent=4)
        logger.info(f"Saved prompts to {output_prompt_path}")

async def generate_themes():
    creator = DatasetCreator()
    previous_themes_path = "themes_with_icons.json"
    output_path = "style_icons_new.json"
    await creator.generate_and_save_style_icons(num_styles=50, previous_themes_path=previous_themes_path, output_path=output_path)


async def generate_prompts():
    creator = PromptCreator()
    themes_path = "style_icons_new.json"
    output_prompt_path = "prompts.json"
    num_icons_to_generate = 10
    await creator.generate_and_save_style_icons(themes_path=themes_path, output_prompt_path=output_prompt_path, num_icons_to_generate=num_icons_to_generate)


if __name__ == "__main__":
    # asyncio.run(generate_themes())
    asyncio.run(generate_prompts())
