import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv('.env')


@dataclass(frozen=True)
class GPTConf:
    model_name: str = 'gpt-4o'
    retries: int = 5
