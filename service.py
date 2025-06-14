import gc
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse

import config
import model_manager as manager
import utils_service
from flux_base import flux_generate, flux_words_lora
from flux_canny import flux_canny
from flux_inpaint import flux_inpaint
from flux_lora import canny_lora, comfy_inf_2, inference_lora
from trajectory import animation_process

app = FastAPI()

model_manager = manager.ModelManager()


@app.post('/flux-generate-image/')
async def flux_generate_image(
    prompt: str = Form(...),
    guidance_scale: float = Form(3.5),
    num_inference_steps: int = Form(28),
    seed: int = Form(0),
    width: int = Form(1024),
    height: int = Form(1024),
) -> FileResponse:
    img = flux_generate.infer(
        prompt=prompt,
        model_manager=model_manager,
        width=width,
        height=height,
        seed=seed,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )
    img_save_path = (
        Path(config.config.dirs.generation_dir) / f'{utils_service.get_hash_from_uuid()}.png'
    )
    img.save(img_save_path)

    return FileResponse(
        path=str(img_save_path),
        media_type='image/jpeg',
        filename=f'processed_{img_save_path.name}',
    )


@app.post('/flux-generate-image-with-tier/')
async def flux_generate_image_with_tier(
    prompt: str = Form(...),
    tier: str = Form(...),
    guidance_scale: float = Form(3.5),
    num_inference_steps: int = Form(28),
    seed: int = Form(0),
    width: int = Form(1024),
    height: int = Form(1024),
    character_lora_strength: float = Form(1.0),
    style_lora_strength: float = Form(0.0),
) -> FileResponse:
    img = flux_generate.infer_with_tier(
        prompt=prompt,
        model_manager=model_manager,
        tier=tier,
        width=width,
        height=height,
        seed=seed,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        character_lora_strength=character_lora_strength,
        style_lora_strength=style_lora_strength,
    )
    img_save_path = (
        Path(config.config.dirs.generation_dir) / f'{utils_service.get_hash_from_uuid()}.png'
    )
    img.save(img_save_path)

    return FileResponse(
        path=str(img_save_path),
        media_type='image/jpeg',
        filename=f'processed_{img_save_path.name}',
    )


@app.post('/flux-generate-image-with-letters-lora/')
async def flux_generate_image_with_lora(
    prompt: str = Form(...),
    lora_name: str = Form(...),
    guidance_scale: float = Form(3.5),
    num_inference_steps: int = Form(28),
    seed: int = Form(0),
    width: int = Form(1024),
    height: int = Form(1024),
    lora_strength: float = Form(1.0),
) -> FileResponse:
    img = flux_words_lora.infer_with_lora(
        prompt=prompt,
        model_manager=model_manager,
        lora_name=lora_name,
        width=width,
        height=height,
        seed=seed,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        lora_strength=lora_strength,
    )
    img_save_path = (
        Path(config.config.dirs.generation_dir) / f'{utils_service.get_hash_from_uuid()}.png'
    )
    img.save(img_save_path)

    return FileResponse(
        path=str(img_save_path),
        media_type='image/jpeg',
        filename=f'processed_{img_save_path.name}',
    )


@app.post('/flux-inpaint-image/')
async def flux_inpaint_image(
    prompt: str = Form(...),
    image: UploadFile = File(...),  # noqa: B008
    image_mask: UploadFile = File(...),  # noqa: B008
    seed: int = Form(0),
    width: int = Form(1024),
    height: int = Form(1024),
    guidance_scale: float = Form(3.5),
    controlnet_conditioning_scale: float = Form(0.4),
    num_inference_steps: int = Form(28),
    true_guidance_scale: float = Form(1.0),
) -> FileResponse:
    base_img_path = (
        Path(config.config.dirs.images_dir)
        / f'{utils_service.get_hash_from_uuid()}{Path(image.filename).suffix}'
    )
    mask_img_path = (
        Path(config.config.dirs.masks_dir)
        / f'{utils_service.get_hash_from_uuid()}{Path(image.filename).suffix}'
    )
    inpainted_img_path = (
        Path(config.config.dirs.inpainting_dir)
        / f'{utils_service.get_hash_from_uuid()}{Path(image.filename).suffix}'
    )

    with base_img_path.open('wb') as file:
        file.write(await image.read())
    with mask_img_path.open('wb') as file:
        file.write(await image_mask.read())

    flux_inpaint.create_inpaint(
        prompt=prompt,
        image_path=str(base_img_path),
        mask_path=str(mask_img_path),
        save_path=str(inpainted_img_path),
        model_manager=model_manager,
        seed=seed,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=num_inference_steps,
        true_guidance_scale=true_guidance_scale,
    )

    # Return the processed file as a response
    return FileResponse(
        path=str(inpainted_img_path),
        media_type='image/jpeg',
        filename=f'processed_{inpainted_img_path.name}',
    )


@app.post('/flux-canny-image/')
async def flux_canny_image(
    prompt: str = Form(...),
    image: UploadFile = File(...),  # noqa: B008
    seed: int = Form(0),
    num_steps: int = Form(50),
    guidance: float = Form(4.0),
    canny_guidance: float = Form(0.7),
) -> FileResponse:
    base_img_path = (
        Path(config.config.dirs.canny_base_dir)
        / f'{utils_service.get_hash_from_uuid()}{Path(image.filename).suffix}'
    )

    canny_img_path = (
        Path(config.config.dirs.canny_dir)
        / f'{utils_service.get_hash_from_uuid()}{Path(image.filename).suffix}'
    )

    with base_img_path.open('wb') as file:
        file.write(await image.read())

    flux_canny.create_canny(
        prompt=prompt,
        control_image_path=base_img_path,
        save_path=str(canny_img_path),
        model_manager=model_manager,
        seed=seed,
        num_steps=num_steps,
        guidance=guidance,
        canny_guidance=canny_guidance,
    )

    # Return the processed file as a response
    return FileResponse(
        path=str(canny_img_path),
        media_type='image/jpeg',
        filename=f'processed_{canny_img_path.name}',
    )


@app.post('/flux-generate-frame-lora/')
async def flux_generate_frame_lora(
    prompt: str = Form(...),
    guidance_scale: float = Form(3.5),
    num_inference_steps: int = Form(28),
    seed: int = Form(0),
    width: int = Form(1024),
    height: int = Form(1024),
) -> FileResponse:
    img = inference_lora.infer(
        prompt=prompt,
        model_manager=model_manager,
        width=width,
        height=height,
        seed=seed,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )
    img_save_path = (
        Path(config.config.dirs.generation_dir) / f'{utils_service.get_hash_from_uuid()}.png'
    )
    img.save(img_save_path)

    return FileResponse(
        path=str(img_save_path),
        media_type='image/jpeg',
        filename=f'processed_{img_save_path.name}',
    )


@app.post('/flux-canny-lora-image/')
async def flux_canny_image(
    prompt: str = Form(...),
    image: UploadFile = File(...),  # noqa: B008
    seed: int = Form(0),
    num_steps: int = Form(30),
    guidance: float = Form(4.0),
    canny_guidance: float = Form(0.7),
) -> FileResponse:
    base_img_path = (
        Path(config.config.dirs.canny_base_dir)
        / f'{utils_service.get_hash_from_uuid()}{Path(image.filename).suffix}'
    )
    canny_img_path = (
        Path(config.config.dirs.canny_dir)
        / f'{utils_service.get_hash_from_uuid()}{Path(image.filename).suffix}'
    )

    with base_img_path.open('wb') as file:
        file.write(await image.read())

    comfy_inf_2.infer(
        prompt=prompt,
        control_image_path=str(base_img_path.resolve()),
        save_path=str(canny_img_path),
        model_manager=model_manager,
        seed=seed,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
    )

    # Return the processed file as a response
    return FileResponse(
        path=str(canny_img_path),
        media_type='image/jpeg',
        filename=f'processed_{canny_img_path.name}',
    )


@app.post('/flux-canny-lora-image-default/')
async def flux_canny_image_default(
    prompt: str = Form(...),
    image: UploadFile = File(...),  # noqa: B008
    seed: int = Form(0),
    num_inference_steps: int = Form(30),
    guidance_scale: float = Form(30.0),
    lora_name: str = Form('reelslora_old'),
    canny_lora_strength: float = Form(0.8),
    frames_lora_strength: float = Form(1.0),
    width: int | None = Form(None),
    height: int | None = Form(None),
) -> FileResponse:
    base_img_path = (
        Path(config.config.dirs.canny_base_dir)
        / f'{utils_service.get_hash_from_uuid()}{Path(image.filename).suffix}'
    )
    canny_img_path = (
        Path(config.config.dirs.canny_dir)
        / f'{utils_service.get_hash_from_uuid()}{Path(image.filename).suffix}'
    )

    with base_img_path.open('wb') as file:
        file.write(await image.read())

    img = canny_lora.inference_frame_canny_with_lora(
        prompt=prompt,
        control_image_path=str(base_img_path.resolve()),
        model_manager=model_manager,
        seed=seed,
        lora_name=lora_name,
        canny_lora_strength=canny_lora_strength,
        frames_lora_strength=frames_lora_strength,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )

    img_save_path = (
        Path(config.config.dirs.generation_dir) / f'{utils_service.get_hash_from_uuid()}.png'
    )
    img.save(img_save_path)

    return FileResponse(
        path=str(img_save_path),
        media_type='image/jpeg',
        filename=f'processed_{img_save_path.name}',
    )


@app.post('/animation_zoom_in/')
async def animation_zoom_in(
    video: UploadFile = File(...),
    zoom_params: str = Form(...),
    sampling_steps: int = Form(...),
) -> FileResponse:
    # Generate unique ID for this request
    model_manager.unload_model()  # free memory

    unique_id = utils_service.get_hash_from_uuid()

    # Create unique directory for this request
    request_dir = Path(config.config.dirs.animation_dir) / unique_id
    request_dir.mkdir(parents=True, exist_ok=True)

    # Set paths for input and output videos in the unique directory
    input_video_path = request_dir / 'base_video.mp4'
    output_video_path = request_dir / 'zoomed.mp4'

    # Save uploaded video
    with input_video_path.open('wb') as file:
        file.write(await video.read())

    # Process the video
    animation_process.process_zoom_in_animation(
        input_video_path, output_video_path, zoom_params=zoom_params, sampling_steps=sampling_steps
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # Return the processed video
    return FileResponse(
        path=str(output_video_path), media_type='video/mp4', filename=f'zoomed_{unique_id}.mp4'
    )


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=11234)
