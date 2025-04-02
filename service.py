from pathlib import Path


import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse

import config
import model_manager as manager
import utils_service
from flux_base import flux_generate
from flux_canny import flux_canny
from flux_inpaint import flux_inpaint
from flux_lora import comfy_inf_2, inference_lora
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


@app.post('/flux-generate-image-with-lora/')
async def flux_generate_image_with_lora(
    prompt: str = Form(...),
    tier: str = Form(...),
    guidance_scale: float = Form(3.5),
    num_inference_steps: int = Form(28),
    seed: int = Form(0),
    width: int = Form(1024),
    height: int = Form(1024),
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


@app.post('/animation_zoom_in/')
async def animation_zoom_in(
    video: UploadFile = File(...),
) -> FileResponse:
    # Generate unique ID for this request
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
    animation_process.process_zoom_in_animation(input_video_path, output_video_path)

    # Return the processed video
    return FileResponse(
        path=str(output_video_path),
        media_type='video/mp4',
        filename=f'zoomed_{unique_id}.mp4'
    )



if __name__ == '__main__':  
    uvicorn.run(app, host='0.0.0.0', port=11234)
