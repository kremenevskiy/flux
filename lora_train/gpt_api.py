import base64
import json
import os
import time
from typing import Any, TypeAlias

import openai
from loguru import logger
from openai import InternalServerError

from lora_train import config, gpt_prompts

ChatCompletionResponse: TypeAlias = openai.types.chat.chat_completion.ChatCompletion


class MaxTokenLimitExceededError(Exception):
    """Exception raised when the maximum token limit is reached."""

    pass


class InvalidGenerationError(Exception):
    """Exception raised when the generated data not as expected."""

    pass


class GptApi:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name if model_name else config.GPTConf.model_name
        self.client = openai.OpenAI(
            api_key=os.environ['OPENAI_KEY'],
            base_url='https://api.proxyapi.ru/openai/v1',
        )

    def _ask_gpt_custom(self, user_prompt: str, system_prompt: str) -> ChatCompletionResponse:
        return self.client.chat.completions.create(
            model=self.model_name,
            response_format={'type': 'json_object'},
            messages=[
                {'role': 'system', 'content': system_prompt},
                {
                    'role': 'user',
                    'content': f'{user_prompt}',
                },
            ],
            max_tokens=1024,
        )

    def _get_json_from_gpt(self, chat_completion: Any) -> dict:
        finish_reason = chat_completion.choices[0].finish_reason
        if finish_reason != 'stop':
            raise MaxTokenLimitExceededError(
                'Max tokens limit reached. Please try again with a shorter prompt.'
            )
        openai_resp = chat_completion.choices[0].message.content
        return json.loads(openai_resp)

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encode an image file to base64 format."""
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_image_description(self, image_path: str) -> str:
        # Encode the image
        base64_image = self._encode_image_to_base64(image_path)

        # Prepare system and user messages
        user_message = {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'Describe this image:'},
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/png;base64,{base64_image}'},
                },
            ],
        }

        # Send the request to GPT
        response = self.client.chat.completions.create(
            model='gpt-4o',  # GPT-4 Vision model
            messages=[
                {
                    'role': 'system',
                    'content': gpt_prompts.system_prompt_for_image_captioning,
                },
                user_message,
            ],
            max_tokens=1024,
        )

        # Parse the response
        return response.choices[0].message.content

    def get_image_description_untill_success(self, image_path: str) -> str:
        retry_minutes = 30
        for attempt in range(1, config.GPTConf.retries + 1):
            try:
                return self.get_image_description(image_path)
            except InternalServerError as e:
                logger.error(
                    f'Attempt {attempt}: InternalServerError - retrying in 1 second. Error: {e}'
                )
                time.sleep(30)
            except Exception as e:
                # Check if the error is due to insufficient balance (Error code 402)
                if '402' in str(e) or 'Insufficient balance' in str(e):
                    logger.error(f'Attempt {attempt}: Insufficient balance error encountered: {e}')
                    if attempt == config.GPTConf.retries:
                        raise RuntimeError(
                            'Insufficient balance to run this request. Stopped Instance'
                        ) from e

                    time.sleep(retry_minutes * 60)
                else:
                    logger.error(f'Unexpected error on attempt {attempt}: {e}')
                    raise
        raise RuntimeError(
            f'Failed to get image description after {config.GPTConf.retries} retries.'
        )
