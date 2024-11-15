from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse

import config
import model_manager as manager
import utils
from flux_base import flux_generate
from flux_inpaint import flux_inpaint

app = FastAPI()

model_manager = manager.ModelManager()


@app.post('/flux-generate-image/')
async def flux_generate_image(
    prompt: str = Form(...),
    seed: int = Form(0),
) -> FileResponse:
    img = flux_generate.infer(prompt=prompt, seed=seed, model_manager=model_manager)
    img_save_path = Path(config.config.dirs.generation_dir) / f'{utils.get_hash_from_uuid}.png'
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
        / f'{utils.get_hash_from_uuid}{Path(image.filename).suffix}'
    )
    mask_img_path = (
        Path(config.config.dirs.masks_dir)
        / f'{utils.get_hash_from_uuid}{Path(image.filename).suffix}'
    )
    inpainted_img_path = (
        Path(config.config.dirs.inpainting_dir)
        / f'{utils.get_hash_from_uuid}{Path(image.filename).suffix}'
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


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=11234)
