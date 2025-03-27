def build_system_prompt(tier: str, style_theme: str, icon_name: str) -> str:
    tier = tier.lower()

    # Tier-based settings
    tier_settings = {
        "top-tier": {
            "style": "Very detailed and ornate, with glowing effects and rich textures.",
            "colors": "Warm golden tones with radiant red and orange accents.",
            "effects": "Sparkles, glow, and shiny highlights.",
            "background": "Stylized fantasy background with ornate details.",
            "value": "Feels legendary and highly valuable.",
            "fill": "100%",
            "color_tone": "warm tones (gold, orange, red)"
        },
        "high-tier": {
            "style": "Polished and refined, but slightly simpler than top-tier.",
            "colors": "Golden tones with subtle warm highlights.",
            "effects": "Soft glow and slight shimmer.",
            "background": "Fantasy-style with light ornamentation.",
            "value": "Looks premium and rare.",
            "fill": "85%",
            "color_tone": "warm tones (gold, orange, red)"
        },
        "mid-tier": {
            "style": "Game-style detail with balanced elegance.",
            "colors": "Gold and green mix, more subdued.",
            "effects": "Minimal glow, soft highlights.",
            "background": "Simple gradients or abstract elements.",
            "value": "Moderately rare and interesting.",
            "fill": "70%",
            "color_tone": "balanced tones (gold, green)"
        },
        "low-mid-tier": {
            "style": "Simpler cartoon-like style.",
            "colors": "Cool tones like turquoise and silver-green.",
            "effects": "No strong glow, smooth finish.",
            "background": "Minimal background with soft color.",
            "value": "Common but nicely designed.",
            "fill": "60%",
            "color_tone": "cool tones (teal, silver)"
        },
        "low-tier": {
            "style": "Flat, clean, minimal design.",
            "colors": "Cool, muted tones like silver or blue.",
            "effects": "No glow, matte finish.",
            "background": "Plain or abstract background.",
            "value": "Very basic icon.",
            "fill": "50%",
            "color_tone": "cool tones (blue, silver)"
        }
    }

    selected = tier_settings.get(tier)
    if not selected:
        raise ValueError("Unknown tier specified.")

    system_prompt = f"""
You are an icon prompt generator for a fantasy-style slot game.

Your task:
Create a short, stylized, game-ready image description (1 sentence) for an icon.
The icon is the central figure in the scene — like a symbol in a slot machine.

Use the following:
- Theme: {style_theme}
- Icon Name: {icon_name}
- Stylistic Tier: {tier.capitalize()}

Visual style (based on tier):
- Style: {selected['style']}
- Colors: {selected['colors']}
- Effects: {selected['effects']}
- Background: {selected['background']}
- Value: {selected['value']}
- Object Focus: fills {selected['fill']} of the image
- Tones: {selected['color_tone']}

Format:
Output a single sentence description that brings the icon to life.
Keep it punchy, vivid, and suitable for game art — no photorealism.

Examples:
- A golden compass surrounded by a mystical glow, lying on an old map with islands and seas marked, and ships and outlines of unexplored lands can be seen in the background.
- A luxurious golden train in retro style, racing along rails sparkling in the sunset light, against a backdrop of picturesque mountain landscapes and bridges stretching into the distance.
- A golden gramophone with a shining bell stands on a wooden table in a cozy vintage room, the walls of which are decorated with paintings and antique clocks, and outside the window there is a view of the night city.


Now create a new prompt:
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