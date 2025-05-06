import os
import random
import sys
from typing import Any, Mapping, Sequence, Union

import torch
import torchvision
from PIL import Image

import model_manager
import utils_service


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj['result'][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f'{name} found: {path_name}')
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path('ComfyUI')
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            'Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead.'
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path('extra_model_paths.yaml')

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print('Could not find the extra_model_paths config file.')


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio

    import execution
    import server
    from nodes import init_extra_nodes

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS


def get_model_pipe():
    dualcliploader = NODE_CLASS_MAPPINGS['DualCLIPLoader']()
    dualcliploader_4 = dualcliploader.load_clip(
        clip_name1='clip_l.safetensors',
        clip_name2='t5xxl_fp16.safetensors',
        type='flux',
        device='default',
    )

    unetloader = NODE_CLASS_MAPPINGS['UNETLoader']()
    unetloader_10 = unetloader.load_unet(unet_name='flux1-dev.safetensors', weight_dtype='default')

    loraloader = NODE_CLASS_MAPPINGS['LoraLoader']()
    loraloader_23 = loraloader.load_lora(
        lora_name='frames_lora_1.safetensors',
        strength_model=0.8,
        strength_clip=1,
        model=get_value_at_index(unetloader_10, 0),
        clip=get_value_at_index(dualcliploader_4, 0),
    )

    cliptextencodeflux = NODE_CLASS_MAPPINGS['CLIPTextEncodeFlux']()
    emptylatentimage = NODE_CLASS_MAPPINGS['EmptyLatentImage']()

    vaeloader = NODE_CLASS_MAPPINGS['VAELoader']()
    vaeloader_8 = vaeloader.load_vae(vae_name='ae.safetensors')

    loadfluxcontrolnet = NODE_CLASS_MAPPINGS['LoadFluxControlNet']()
    loadfluxcontrolnet_13 = loadfluxcontrolnet.loadmodel(
        model_name='flux-dev',
        controlnet_path='flux-canny-controlnet-v3.safetensors',
    )

    loadimage = NODE_CLASS_MAPPINGS['LoadImage']()

    cannyedgepreprocessor = NODE_CLASS_MAPPINGS['CannyEdgePreprocessor']()
    applyfluxcontrolnet = NODE_CLASS_MAPPINGS['ApplyFluxControlNet']()
    xlabssampler = NODE_CLASS_MAPPINGS['XlabsSampler']()
    vaedecode = NODE_CLASS_MAPPINGS['VAEDecode']()

    return (
        cliptextencodeflux,
        loraloader_23,
        emptylatentimage,
        loadimage,
        cannyedgepreprocessor,
        applyfluxcontrolnet,
        loadfluxcontrolnet_13,
        xlabssampler,
        vaedecode,
        vaeloader_8,
        dualcliploader_4,
    )


def infer(
    prompt: str,
    control_image_path: str,
    save_path: str,
    model_manager: model_manager.ModelManager,
    seed: int = 42,
    randomize_seed: bool = True,
    width: int | None = None,
    height: int | None = None,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 20,
) -> torch.Tensor:
    control_image = Image.open(control_image_path)
    control_image = utils_service.resize_to_nearest_multiple(image=control_image)
    control_image.save(control_image_path)

    width, height = control_image.size if (width is None and height is None) else (width, height)
    print('set width and height: ', width, height)

    import_custom_nodes()
    with torch.inference_mode():
        (
            cliptextencodeflux,
            loraloader_23,
            emptylatentimage,
            loadimage,
            cannyedgepreprocessor,
            applyfluxcontrolnet,
            loadfluxcontrolnet_13,
            xlabssampler,
            vaedecode,
            vaeloader_8,
            dualcliploader_4,
        ) = model_manager.get_model(model_name='flux_lora_canny')

        cliptextencodeflux_5 = cliptextencodeflux.encode(
            clip_l=prompt,
            t5xxl=prompt,
            guidance=4,
            clip=get_value_at_index(loraloader_23, 1),
        )

        emptylatentimage_6 = emptylatentimage.generate(width=width, height=height, batch_size=1)

        loadimage_16 = loadimage.load_image(image=control_image_path)

        cliptextencodeflux_19 = cliptextencodeflux.encode(
            clip_l='',
            t5xxl='',
            guidance=4,
            clip=get_value_at_index(dualcliploader_4, 0),
        )

        for q in range(1):
            cannyedgepreprocessor_15 = cannyedgepreprocessor.execute(
                low_threshold=100,
                high_threshold=200,
                resolution=832,
                image=get_value_at_index(loadimage_16, 0),
            )

            img_pt = get_value_at_index(cannyedgepreprocessor_15, 0)
            img_pt = img_pt.squeeze(0)
            img_pt = img_pt.permute(2, 0, 1)
            torchvision.utils.save_image(img_pt, 'canny_edge.png')

            applyfluxcontrolnet_14 = applyfluxcontrolnet.prepare(
                strength=0.6000000000000001,
                controlnet=get_value_at_index(loadfluxcontrolnet_13, 0),
                image=get_value_at_index(cannyedgepreprocessor_15, 0),
            )

            xlabssampler_3 = xlabssampler.sampling(
                noise_seed=seed,
                steps=num_inference_steps,
                timestep_to_start_cfg=1,
                true_gs=guidance_scale,
                image_to_image_strength=0,
                denoise_strength=1,
                model=get_value_at_index(loraloader_23, 0),
                conditioning=get_value_at_index(cliptextencodeflux_5, 0),
                neg_conditioning=get_value_at_index(cliptextencodeflux_19, 0),
                latent_image=get_value_at_index(emptylatentimage_6, 0),
                controlnet_condition=get_value_at_index(applyfluxcontrolnet_14, 0),
            )

            vaedecode_7 = vaedecode.decode(
                samples=get_value_at_index(xlabssampler_3, 0),
                vae=get_value_at_index(vaeloader_8, 0),
            )

            img_pt = get_value_at_index(vaedecode_7, 0)
            img_pt = img_pt.squeeze(0)
            img_pt = img_pt.permute(2, 0, 1)
            torchvision.utils.save_image(img_pt, save_path)
            del vaedecode_7
            del xlabssampler_3
            del applyfluxcontrolnet_14
            del cannyedgepreprocessor_15
            del cliptextencodeflux_19
            del loadimage_16
            del emptylatentimage_6
            del cliptextencodeflux_5
            torch.cuda.empty_cache()
            return img_pt
    return


if __name__ == '__main__':
    sv_path = '/root/test_im.png'
    cntrl_pth = '/root/flux/data/canny-base/0facd.png'
    infer(
        prompt='Create a rectangular frame for a slot machine in a steampunk style. The frame should be decorated with intricate gears, cogs, pipes, and steam vents, along with metallic textures like brass, copper, and iron. Incorporate Victorian-inspired design elements such as ornate patterns, rivets, clocks, and goggles. Add subtle details like glowing tubes, pressure gauges, and vintage levers for a mechanical and industrial aesthetic. The frame should be isolated on a plain white background, with the area both inside and outside of the frame completely white',
        control_image_path=cntrl_pth,
        save_path=sv_path,
        width=None,
        height=None,
    )
