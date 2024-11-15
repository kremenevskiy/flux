import random

import gradio as gr
import numpy as np

# import spaces
import torch
from diffusers import (
    AutoencoderKL,
    AutoencoderTiny,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
)
from live_preview_helpers import (
    calculate_shift,
    flux_pipe_call_that_returns_an_iterable_of_images,
    retrieve_timesteps,
)
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

taef1 = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=dtype).to(
    device
)
good_vae = AutoencoderKL.from_pretrained(
    "black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=dtype
).to(device)
pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=dtype, vae=taef1
).to(device)
torch.cuda.empty_cache()

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

pipe.flux_pipe_call_that_returns_an_iterable_of_images = (
    flux_pipe_call_that_returns_an_iterable_of_images.__get__(pipe)
)


# @spaces.GPU(duration=75)
def infer(
    prompt,
    seed=42,
    randomize_seed=False,
    width=1024,
    height=1024,
    guidance_scale=3.5,
    num_inference_steps=28,
    progress=gr.Progress(track_tqdm=True),
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)

    for img in pipe.flux_pipe_call_that_returns_an_iterable_of_images(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=generator,
        output_type="pil",
        good_vae=good_vae,
    ):
        yield img, seed


examples = [
    "a tiny astronaut hatching from an egg on the moon",
    "a cat holding a sign that says hello world",
    "an anime illustration of a wiener schnitzel",
]

css = """
#col-container {
    margin: 0 auto;
    max-width: 520px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""# FLUX.1 [dev]
12B param rectified flow transformer guidance-distilled from [FLUX.1 [pro]](https://blackforestlabs.ai/)  
[[non-commercial license](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)] [[blog](https://blackforestlabs.ai/announcing-black-forest-labs/)] [[model](https://huggingface.co/black-forest-labs/FLUX.1-dev)]
        """)

        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )

            run_button = gr.Button("Run", scale=0)

        result = gr.Image(label="Result", show_label=False)

        with gr.Accordion("Advanced Settings", open=False):
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )

                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )

            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1,
                    maximum=15,
                    step=0.1,
                    value=3.5,
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=28,
                )

        gr.Examples(
            examples=examples,
            fn=infer,
            inputs=[prompt],
            outputs=[result, seed],
            cache_examples="lazy",
        )

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[
            prompt,
            seed,
            randomize_seed,
            width,
            height,
            guidance_scale,
            num_inference_steps,
        ],
        outputs=[result, seed],
    )

demo.launch()
