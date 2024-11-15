from dataclasses import dataclass
import os


@dataclass(frozen=True)
class DirsConfig:
    iamges_dir: str = 'data/inpaint-images'
    masks_dir: str = 'data/inpaint-masks'
    inpainted_dir: str = 'data/inpaint'
    generated_dir: str = 'data/generated-images'



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
