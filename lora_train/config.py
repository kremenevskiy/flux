import os
from dataclasses import dataclass

from dotenv import load_dotenv


if os.path.exists('/root/flux/.env'):
    load_dotenv('/root/flux/.env')
else:
    raise FileNotFoundError('No .env file found')


@dataclass(frozen=True)
class GPTConf:
    model_name: str = 'gpt-4o'
    retries: int = 5
