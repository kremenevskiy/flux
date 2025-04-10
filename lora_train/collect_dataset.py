from pathlib import Path

from loguru import logger

from lora_train import gpt_api


class DatasetProcessor:
    def __init__(self, trigger_word: str | None = None, put_only_trigger_word: bool = False):
        self.trigger_word = trigger_word
        self.put_only_trigger_word = put_only_trigger_word
        assert self.put_only_trigger_word or self.trigger_word is not None, (
            'trigger_word must be provided if put_only_trigger_word is True'
        )

    def process_dataset(self) -> None:
        self.generate_descriptions_for_local_photos()

    async def generate_photo_description(self, photo_path: str) -> str:
        if self.put_only_trigger_word:
            return self.trigger_word
        image_description = await gpt_api.GptApi().get_image_description_untill_success(
            image_path=photo_path
        )
        if self.trigger_word:
            image_description = f'{self.trigger_word}, {image_description}'
        return image_description

    async def generate_descriptions_for_local_photos(self, photos_dir: str, output_dir: str) -> None:
        photos_dir_pth = Path(photos_dir)
        output_dir_pth = Path(output_dir)
        output_dir_pth.mkdir(parents=True, exist_ok=True)
        if not photos_dir_pth.exists() or not photos_dir_pth.is_dir():
            logger.error(f'The directory {photos_dir} does not exist or is not a directory.')
            return

        for idx, photo_path in enumerate(photos_dir_pth.glob('*'), start=1):
            if photo_path.is_file() and photo_path.suffix.lower() in [
                '.jpg',
                '.jpeg',
                '.png',
            ]:
                logger.info(f'Processing: {photo_path.name}')

                # Generate description
                text_save_path = output_dir_pth / f'{idx}.txt'
                if text_save_path.exists():
                    logger.info(f'Description already exists: {text_save_path}')
                    continue
                image_description = await self.generate_photo_description(photo_path=str(photo_path))

                # Save description in a text file

                with text_save_path.open('w', encoding='utf-8') as text_file:
                    text_file.write(image_description)

                image_save_path = output_dir_pth / f'{idx}{photo_path.suffix}'
                with (
                    open(photo_path, 'rb') as image_file,
                    open(image_save_path, 'wb') as output_image_file,
                ):
                    output_image_file.write(image_file.read())

                logger.info(f'Processed: {photo_path.name} | Description saved: {text_save_path}')

import asyncio

async def main():
    ds_processor = DatasetProcessor(trigger_word='pic_a, luxurious style, gold, high importance, ultra close up')
    ds_path = '/root/data/PicA_New'
    output_path = '/root/data/pic_a/'
    await ds_processor.generate_descriptions_for_local_photos(photos_dir=ds_path, output_dir=output_path)

if __name__ == '__main__':
    asyncio.run(main())
