import json
import os
from typing import TypeAlias

import openai
from dotenv import load_dotenv
from loguru import logger

from symbols_priority.icons_priority import icons_prompt

load_dotenv('.env', override=True)

ChatCompletionResponse: TypeAlias = openai.types.chat.chat_completion.ChatCompletion


class MaxTokenLimitExceededError(Exception):
    """Exception raised when the maximum token limit is reached."""


class NoMesssageInCompletionError(Exception):
    """Exception raised when the maximum token limit is reached."""


class InvalidGenerationError(Exception):
    """Exception raised when the generated data not as expected."""


class GptApi:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name if model_name else 'gpt-4o-mini'
        self.client = openai.OpenAI(
            api_key=os.environ['OPENAI_PROXY_KEY'],
            base_url='https://api.proxyapi.ru/openai/v1',
        )

    def _ask_gpt_custom(
        self, user_prompt: str, system_prompt: str, model_name: str | None = None
    ) -> ChatCompletionResponse:
        logger.info(f'Asking gpt with model: {model_name}')
        return self.client.chat.completions.create(
            model=model_name if model_name else self.model_name,
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

    def _get_json_from_gpt(self, chat_completion: ChatCompletionResponse) -> dict:
        finish_reason = chat_completion.choices[0].finish_reason
        if finish_reason != 'stop':
            raise MaxTokenLimitExceededError(
                'Max tokens limit reached. Please try again with a shorter prompt.'
            )

        openai_resp = chat_completion.choices[0].message.content
        if openai_resp is None:
            raise NoMesssageInCompletionError

        # return generated config
        return json.loads(openai_resp)

    def get_sorted_symbols(self, theme: str, model_name: str | None = None) -> dict:
        user_prompt = f'Theme: {theme}'
        model_name = model_name if model_name else self.model_name
        chat_completion = self._ask_gpt_custom(
            user_prompt=user_prompt,
            system_prompt=icons_prompt.ICONS_SYSTEM_PROMPT,
            model_name=model_name,
        )
        return self._get_json_from_gpt(chat_completion=chat_completion)
