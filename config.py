import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DirsConfig:
    images_dir: str = 'data/inpaint-images'
    masks_dir: str = 'data/inpaint-masks'
    inpainting_dir: str = 'data/inpaint'
    generation_dir: str = 'data/generation-images'


@dataclass(frozen=True)
class Config:
    dirs: DirsConfig


def create_dirs(dirs: DirsConfig):
    """
    Create directories if they do not exist.
    """
    for dir_path in vars(dirs).values():
        os.makedirs(dir_path, exist_ok=True)


def load_config():
    create_dirs(DirsConfig())
    config = Config(dirs=DirsConfig())
    return config


config = load_config()
