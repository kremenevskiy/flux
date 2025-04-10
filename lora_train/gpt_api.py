import base64
import json
import os
import time
from typing import Any, TypeAlias

import openai
from loguru import logger
from openai import InternalServerError
from openai.types.chat import ChatCompletion

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
        self.client = openai.AsyncOpenAI(
            api_key=os.environ['OPENAI_PROXY_KEY'],
            base_url='https://api.proxyapi.ru/openai/v1',
        )

    async def _create_chat_completion(
        self,
        system_prompt: str,
        user_messages: list[dict],
        max_tokens: int = 1024,
        response_format: dict | None = None,
    ) -> ChatCompletion:
        messages = [{"role": "system", "content": system_prompt}] + user_messages
        return await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            **({"response_format": response_format} if response_format else {}),
        )
    
    async def _ask_gpt(
        self, user_prompt: str, system_prompt: str
    ) -> ChatCompletion:
        user_messages = [{"role": "user", "content": user_prompt}]
        return await self._create_chat_completion(
            system_prompt=system_prompt,
            user_messages=user_messages,
            response_format={"type": "json_object"},
        )
       
    def _get_json_from_gpt(self, chat_completion: ChatCompletion) -> dict:
        finish_reason = chat_completion.choices[0].finish_reason
        openai_resp = chat_completion.choices[0].message.content
        if finish_reason != "stop":
            raise MaxTokenLimitExceededError(
                f"Max tokens limit reached. Please try again with a shorter prompt. Prompt: {openai_resp}"
            )

        return parse_json_from_gpt(response_str=openai_resp)
    

    async def ask_gpt(
        self, user_input: str, system_prompt: str,
    ) -> str:
        chat_completion = await self._ask_gpt(
            user_prompt=user_input,
            system_prompt=system_prompt,
        )
        return self._get_json_from_gpt(chat_completion=chat_completion)
    

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encode an image file to base64 format."""
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    async def get_image_description(self, image_path: str) -> str:
        base64_image = self._encode_image_to_base64(image_path)
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
        response = await self.client.chat.completions.create(
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


def parse_json_from_gpt(response_str: str) -> dict:
    response_str = response_str.strip()
    try:
        return json.loads(response_str)
    except json.JSONDecodeError:
        # Check for double curly braces: {{ ... }}
        if response_str.startswith("{{") and response_str.endswith("}}"):
            fixed_response = response_str[1:-1]
            try:
                return json.loads(fixed_response)
            except json.JSONDecodeError as e2:
                logger.error("Failed to parse GPT response as JSON:")
                logger.error(f"Bad Response: {fixed_response}")
                logger.error(f"Exception: {e2}")
                raise ValueError("Invalid JSON from GPT response") from e2