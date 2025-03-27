def build_system_prompt(tier: str, style_theme: str, icon_name: str) -> str:
    tier = tier.lower()

    # Tier-based settings
    tier_settings = {
        "top-tier": {
            "style": "Extremely detailed, luxurious, with intricate patterns and fine textures.",
            "colors": "Dominantly golden with radiant highlights, complemented by rich red and orange accents.",
            "effects": "Dynamic lighting with radiant glows, sparkles, and polished reflections.",
            "background": "Highly ornate with premium embellishments, such as filigree or embedded gems.",
            "value": "Exudes grandeur and exclusivity, with an unmistakably expensive feel.",
            "fill": "100%",
            "color_tone": "warm tones (orange, golden, red)"
        },
        "high-tier": {
            "style": "Very refined but slightly less intricate.",
            "colors": "Warm golden tone with subtle red or orange highlights.",
            "effects": "Gentle glows and softer lighting effects.",
            "background": "Decorated but simpler, with smaller patterns or symbols.",
            "value": "Still premium, though less overwhelming in its opulence.",
            "fill": "85%",
            "color_tone": "warm tones (orange, golden, red)"
        },
        "mid-tier": {
            "style": "Moderately detailed with a balance of elegance and simplicity.",
            "colors": "A mix of gold and green, with a shift toward earthy tones.",
            "effects": "Minimal glows or reflections, muted highlights.",
            "background": "Smooth gradients or subtle textures.",
            "value": "Appears valuable but lacks elite refinement.",
            "fill": "70%",
            "color_tone": "balanced tones (gold, green)"
        },
        "low-mid-tier": {
            "style": "Simpler design with fewer textures and decorative patterns.",
            "colors": "Cool tones such as teal, turquoise, or silvery-green.",
            "effects": "Soft, diffused lighting with no significant glow.",
            "background": "Minimalistic, basic gradients or textures.",
            "value": "Functional but less ornate.",
            "fill": "60%",
            "color_tone": "cool tones (teal, silver, turquoise)"
        },
        "low-tier": {
            "style": "Simplistic with almost no texture or ornamentation.",
            "colors": "Cool, subdued tones like silver or light blue.",
            "effects": "Flat or matte finish with no glow or lighting effects.",
            "background": "Plain and unobtrusive, no embellishments.",
            "value": "Basic, utilitarian, and clearly lower-tier.",
            "fill": "50%",
            "color_tone": "cool tones (blue, silver)"
        }
    }

    selected = tier_settings.get(tier)
    if not selected:
        raise ValueError("Unknown tier specified.")

    system_prompt = f"""
You are a prompt generator that produces visually rich, fantasy-style image generation prompts for AI models like MidJourney, DALL·E, or Stable Diffusion.

You will be given:
- A visual theme or style (e.g., “Mystic”, “Cyberpunk”, “Harry Potter”)
- An icon name (e.g., “Golden Compass”, “Magic Hat”)
- A stylistic tier (e.g., “Top-tier”, “Mid-tier”, etc.)

🎨 Your task:
Create a highly detailed and vivid image prompt describing:
- The central icon as the **main figure** or **hero** of the image.
- What is visually happening around it.
- A brief sense of the background or setting.

The image should feel **alive**, **expressive**, and **immersive**, like key art for a fantasy game. Your tone should feel cinematic and rich in atmosphere.

🏆 Follow these visual rules based on stylistic tier:
- Style & Detail: {selected['style']}
- Color Palette: {selected['colors']}
- Effects: {selected['effects']}
- Background: {selected['background']}
- Value & Prestige: {selected['value']}
- Object Focus: Fill {selected['fill']} of the image.
- Color Tone: Emphasize {selected['color_tone']}.

📦 Output: A single paragraph prompt, no bullet points or tags.

📚 Examples:

- A golden compass surrounded by a mystical glow, lying on an old map with islands and seas marked, and ships and outlines of unexplored lands can be seen in the background.

- A luxurious golden train in retro style, racing along rails sparkling in the sunset light, against a backdrop of picturesque mountain landscapes and bridges stretching into the distance.

- A golden gramophone with a shining bell stands on a wooden table in a cozy vintage room, the walls of which are decorated with paintings and antique clocks, and outside the window there is a view of the night city.

Now generate a unique image prompt for the icon:

Theme: {style_theme}  
Icon Name: {icon_name}  
Stylistic Tier: {tier.capitalize()}
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