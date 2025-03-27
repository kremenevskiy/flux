def build_system_prompt(tier: str, style_theme: str, icon_name: str) -> str:
    tier = tier.lower()

    tier_settings = {
        "top-tier": {
            "style": "Very detailed and ornate, with glowing effects and rich textures.",
            "colors": "Warm golden tones with radiant red and orange accents.",
            "effects": "Sparkles, glow, and shiny highlights.",
            "background": "Stylized fantasy background with ornate details.",
            "value": "Feels legendary and highly valuable.",
            "fill": "100%",
            "color_tone": "warm tones (gold, orange, red)",
            "layers": "Foreground (main icon), midground (decorative elements), background (distant scene)",
            "size": "The icon should be very large and dominant, filling almost the entire frame."
        },
        "high-tier": {
            "style": "Polished and refined, but slightly simpler than top-tier.",
            "colors": "Golden tones with subtle warm highlights.",
            "effects": "Soft glow and slight shimmer.",
            "background": "Fantasy-style with light ornamentation.",
            "value": "Looks premium and rare.",
            "fill": "85%",
            "color_tone": "warm tones (gold, orange, red)",
            "layers": "Foreground (main icon), midground (supporting details), background (light stylization)",
            "size": "The icon should be large and clearly central."
        },
        "mid-tier": {
            "style": "Game-style detail with balanced elegance.",
            "colors": "Gold and green mix, more subdued.",
            "effects": "Minimal glow, soft highlights.",
            "background": "Simple gradients or abstract elements.",
            "value": "Moderately rare and interesting.",
            "fill": "70%",
            "color_tone": "balanced tones (gold, green)",
            "layers": "Foreground (main icon), background (simple pattern or color gradient)",
            "size": "The icon should be medium-sized and balanced within the frame."
        },
        "low-mid-tier": {
            "style": "Simpler cartoon-like style.",
            "colors": "Cool tones like turquoise and silver-green.",
            "effects": "No strong glow, smooth finish.",
            "background": "Minimal background with soft color.",
            "value": "Common but nicely designed.",
            "fill": "60%",
            "color_tone": "cool tones (teal, silver)",
            "layers": "Foreground (icon), background (flat or minimal design)",
            "size": "The icon should be small to medium-sized with space around it."
        },
        "low-tier": {
            "style": "Flat, clean, minimal design.",
            "colors": "Cool, muted tones like silver or blue.",
            "effects": "No glow, matte finish.",
            "background": "Plain or abstract background.",
            "value": "Very basic icon.",
            "fill": "50%",
            "color_tone": "cool tones (blue, silver)",
            "layers": "Foreground (icon), background (plain color or pattern)",
            "size": "The icon should be small and centered, with plenty of empty space."
        }
    }

    selected = tier_settings.get(tier)
    if not selected:
        raise ValueError("Unknown tier specified.")

    system_prompt = f"""
You are an icon prompt generator for a fantasy-style slot game.

Task:
Generate a **one-sentence** prompt describing a stylized icon as it appears in a slot game. The image must be clearly layered and composed for visual clarity and appeal.

Key Instructions:
- The icon is the central figure of the image.
- Break the scene into 2–3 **visual layers**: foreground (icon), midground (decorative elements), background (distant or simple setting). Use fewer layers for lower-tier icons.
- Ensure the **main icon contrasts** with the background — it must not blend in. Background tones should differ.
- Use a **{selected['size']}**
- The style should be **{selected['style']}**
- Use **{selected['colors']}**
- Effects: **{selected['effects']}**
- Background: **{selected['background']}**
- Visual Value: **{selected['value']}**
- Filling: **{selected['fill']} of the image**
- Overall Tones: **{selected['color_tone']}**

Layering: {selected['layers']}

Format:
Output a **single, punchy sentence** describing the icon in its slot-style fantasy scene.

Examples:
- A golden compass glowing above a mystical map, with shimmering islands in the midground and ancient scrolls fading in the background.
- A retro train gleaming in the foreground, rails curving through spark-lit cliffs, and a golden sunset stretching across the distant skyline.
- A shining gramophone on a wooden table, with candlelight flickering in the middle and an old city through the window behind it.

And main:
Image should have an active visual interaction with the viewer.

Now generate a new image prompt:
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