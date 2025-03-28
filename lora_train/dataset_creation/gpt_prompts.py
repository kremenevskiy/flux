def build_system_prompt(tier: str, style_theme: str, icon_name: str) -> str:
    tier = tier.lower()
    tier_settings = {
        "top-tier": {
            "description": (
                "A massive, dominant icon drenched in radiant, opulent golden hues "
                "with dazzling glows and rich ornamentation that screams jackpot, "
                "occupying almost the entire frame."
            )
        },
        "high-tier": {
            "description": (
                "A large, eye-catching icon with warm golden accents and intricate details, "
                "vibrant and premium yet slightly less grand than a jackpot display."
            )
        },
        "mid-tier": {
            "description": (
                "A balanced, moderate-sized icon with clear details and a mix of warm neutral tones, "
                "ensuring strong visual impact while leaving room for layered background elements."
            )
        },
        "low-tier": {
            "description": (
                "A small, modest icon styled in cool, icy blue tones with minimal ornamentation, "
                "subtly set against a clean, contrasting backdrop."
            )
        }
    }

    selected = tier_settings.get(tier)
    if not selected:
        raise ValueError("Unknown tier specified.")

    system_prompt = f"""
You are a slot game icon prompt generator.

Task:
Generate a **one-sentence** prompt that vividly describes a stylized slot game icon in a layered scene.
The image must clearly separate the **foreground (the icon)**, a subtle **midground (decorative elements)**, 
and a simple **background** to enhance clarity and appeal.

Guidelines:
- Theme: {style_theme}
- Icon Name: {icon_name}
- Tier: {tier.capitalize()}
- {selected['description']}
- The icon must be distinct from its background with clear contrast.
- Use concise, vivid language to evoke the look and feel of a slot game reward.

Now generate the prompt sentence.
"""

    return system_prompt.strip()



def build_theme_generator_prompt(previous_themes: list[dict]) -> str:
    

    return f"""
You are a creative generator of unique stylistic themes and matching icon ideas for image generation systems.

Your task is to come up with a **completely original and imaginative visual theme** and list 5 to 10 **icon ideas** that would naturally belong in that theme. The icons should be visually distinct, rich in concept, and all aligned with the generated theme.

🎯 Your goal:
- Avoid all repetition. Do **not** generate any theme that overlaps conceptually or by name with the following themes:
{previous_themes}

🧠 Guidelines:
- The theme must be expressive, with strong visual identity. Example structure: "Mystic Cyberpunk", "Frozen Samurai Realms", "Underworld Circus", "Neon Jungle Spirits".
- Avoid generic names like “Fantasy”, “Magic”, “Animals” — be specific, fresh, cinematic.
- Icons should include objects, characters, symbols, or artifacts related to the theme.
- All icon names should be imaginative and 1–3 words long.

📦 Respond strictly in this JSON format:

{{
  "theme": "<your new theme name here>",
  "icons": [
    "<icon 1>",
    "<icon 2>",
    "...",
    "<icon N>"
  ]
}}

⚠️ Important:
- Never repeat or remix previously listed themes.
- Do not explain your reasoning or include anything outside the JSON.
- Always invent something fresh and visually exciting, like a concept art universe.
""".strip()


if __name__ == "__main__":
    print(build_system_prompt(tier="top-tier", style_theme="Mystic Cyberpunk", icon_name="Golden Compass"))