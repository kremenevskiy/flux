def build_system_prompt(tier: str, style_theme: str, icon_name: str) -> str:
    tier = tier.lower()
    tier_settings = {
        "top-tier": {
            "description": (
                "A top-tier icon with exquisite, luxurious detail and intricate textures, rendered in dominant, radiant golden hues "
                "with vibrant red and orange accents. It features dynamic glows, sparkles, and polished reflections against an ornate backdrop "
                "of filigree and gems, occupying 90-100% of the image to serve as the primary symbol in its group."
            )
        },
        "high-tier": {
            "description": (
                "A high-tier icon with refined detail and elegant textures, rendered in warm tones of pink, orange, or red with subtle highlights. "
                "It boasts gentle glows and soft reflections on a tastefully decorated background, occupying around 85% of the image for a premium look."
            )
        },
        "mid-tier": {
            "description": (
                "A mid-tier icon with balanced detail and graceful design, rendered in warm hues of pink, orange, or red. "
                "It maintains a strong visual presence while being approximately 30% smaller than the top-tier symbol, occupying about 70% of the image."
            )
        },
        "low-mid-tier": {
            "description": (
                "A low-mid-tier icon with attractive design and clear detail, rendered in cool tones such as green or blue (teal or turquoise) with tasteful accents, "
                "occupying around 60% of the image to denote its lower hierarchy while remaining visually engaging."
            )
        },
        "low-tier": {
            "description": (
                "A low-tier icon with a beautifully crafted design in cool, subdued tones like silver or light blue, featuring a graceful finish and a clean backdrop, "
                "occupying about 50% of the image to clearly indicate its lesser significance while still being appealing."
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
- Ensure the icon is centered and contrasts distinctly with its background.
- All icons must be visually appealing; differences lie solely in color and size.
- Rendering Details:
  - For top-tier: The primary symbol in its group, rendered with luxurious, opulent elements, abundant gold and lavish details, multicolored, filling 90-100% of the image.
  - For high-tier and mid-tier: Use warm colors such as pink, orange, and red.
  - For low-mid-tier and low-tier: Use cool colors such as green and blue.
  - Visual interaction with the viewer is essential, especially if the symbol represents an animal or a human.

Examples of good prompts:
- A golden compass surrounded by a mystical glow, lying on an old map with islands and seas marked, while ships and outlines of unexplored lands appear in the background.
- A luxurious golden train in retro style, racing along rails sparkling in the sunset light, against a backdrop of picturesque mountain landscapes and distant bridges.
- A golden gramophone with a shining bell stands on a wooden table in a cozy vintage room, with paintings and antique clocks adorning the walls and a night city visible through the window.

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