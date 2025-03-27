import json
import logging
import os
import openai
from openai.types.chat import ChatCompletion

from . import gpt_prompts

from dotenv import load_dotenv

load_dotenv('.env', override=True)


class MaxTokenLimitExceededError(Exception):
    """Exception raised when the maximum token limit is reached."""


class GptApi:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, model_name: str | None = None):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self.model_name = model_name if model_name else 'gpt-4o-mini'
        self.client_proxy = openai.AsyncOpenAI(
            api_key=os.environ["OPENAI_PROXY_KEY"],
            base_url="https://api.proxyapi.ru/openai/v1",
        )
        self._initialized = True

    async def _create_chat_completion(
        self,
        system_prompt: str,
        user_messages: list[dict],
        max_tokens: int = 1024,
        response_format: dict | None = None,
    ) -> ChatCompletion:
        messages = [{"role": "system", "content": system_prompt}] + user_messages
        return await self.client_proxy.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            **({"response_format": response_format} if response_format else {}),
        )

    async def ask_gpt(
        self, user_prompt: str, system_prompt: str, return_json: bool = True
    ) -> ChatCompletion:
        user_messages = [{"role": "user", "content": user_prompt}]
        chat_completion = await self._create_chat_completion(
            system_prompt=system_prompt,
            user_messages=user_messages,
            max_tokens=1024,
            response_format={"type": "json_object"} if return_json else None,
        )
        return self.get_completion_content(chat_completion, parse_json=return_json)

    def get_completion_content(self, chat_completion: ChatCompletion, parse_json: bool = True) -> dict:
        finish_reason = chat_completion.choices[0].finish_reason
        openai_resp = chat_completion.choices[0].message.content
        if finish_reason != "stop":
            raise MaxTokenLimitExceededError(
                f"Max tokens limit reached. Please try again with a shorter prompt. Prompt: {openai_resp}"
            )

        return parse_json_from_gpt(response_str=openai_resp) if parse_json else openai_resp

    async def get_prompt_from_image(self, image: str, central_object: str) -> str:
        if self.is_test:
            return "aboba image"
        image_description = await self.get_image_description(image=image)
        return await self.get_changed_image_prompt(
            image_prompt=image_description, central_object=central_object
        )

    async def get_image_description(self, image: str) -> str | None:
        # Prepare system and user messages
        system_prompt = "You are a highly advanced AI capable of analyzing images and providing detailed, vivid descriptions. Your descriptions must be creative, visual, and immersive, capturing all essential details of the image."
        user_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image}"},
                    },
                ],
            }
        ]
        response = await self._create_chat_completion(
            system_prompt=system_prompt,
            user_messages=user_messages,
        )
        return response.choices[0].message.content


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
                logging.error("Failed to parse GPT response as JSON:")
                logging.error(f"Bad Response: {fixed_response}")
                logging.error(f"Exception: {e2}")
                raise ValueError("Invalid JSON from GPT response") from e2
