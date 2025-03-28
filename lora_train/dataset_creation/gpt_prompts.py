def build_system_prompt(tier: str, style_theme: str, icon_name: str) -> str:
    tier = tier.lower()
    tier_settings = {
        "top-tier": {
            "description": (
                "A striking icon with exquisite detail and luxurious textures, rendered in dominant, radiant golden hues "
                "with vibrant red and orange accents, dynamic glows, sparkles, and polished reflections against an ornate backdrop of filigree and gems, "
                "occupying 100% of the image to exude grandeur and exclusivity."
            )
        },
        "high-tier": {
            "description": (
                "A refined icon with elegant detail and subtle intricacies, rendered in warm bronze tones with delicate red or orange highlights, "
                "showcasing gentle glows and soft lighting effects on a simpler patterned background, occupying 85% of the image for a premium look."
            )
        },
        "mid-tier": {
            "description": (
                "A balanced icon with moderate detail and a blend of green and earthy hues, featuring minimal glows and muted reflections on a smooth gradient backdrop, "
                "occupying 70% of the image for clear visual impact."
            )
        },
        "low-mid-tier": {
            "description": (
                "A simpler icon with minimal textures and decorative patterns, rendered in cool tones like teal, turquoise, or silvery-green, "
                "with soft, diffused lighting and a minimalistic background of basic gradients, occupying 60% of the image for a functional yet appealing look."
            )
        },
        "low-tier": {
            "description": (
                "A minimalistic icon with almost no ornamentation, rendered in cool, subdued tones such as silver or light blue, "
                "with a flat or matte finish and an unobtrusive plain background, occupying 50% of the image to clearly denote lower significance."
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
- Ensure the icon contrasts distinctly with its background.
- Use concise, vivid language to evoke the look and feel of a slot game icon.

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