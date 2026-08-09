import os
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
import torch

@dataclass
class Config:
    seed: int = 42
    debug: bool = False
    
    input_dir: Path = Path("/kaggle/input/rsna-2026-knee-abnormality-detection")
    output_dir: Path = Path("/kaggle/working/artifacts")
    train_csv: Path = field(init=False)
    series_csv: Path = field(init=False)
    train_images_dir: Path = field(init=False)

    image_size: Tuple[int, int] = (384, 384)
    num_slices: int = 5
    in_channels: int = 15
    
    model_name: str = "convnext_small.fb_in22k_ft_in1k_384"
    pretrained: bool = True
    num_classes: int = 12
    drop_rate: float = 0.2
    drop_path_rate: float = 0.1

    epochs: int = 12
    batch_size: int = 16
    accum_steps: int = 2
    lr: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-2
    num_workers: int = 4
    use_amp: bool = True
    
    target_columns: List[str] = field(default_factory=lambda: [
        "acl_tear",
        "mcl_tear",
        "lcl_tear",
        "pcl_tear",
        "meniscal_tear",
        "cartilage_loss",
        "osteoarthritis",
        "bone_marrow_lesion",
        "fracture",
        "effusion",
        "synovitis",
        "baker_cyst"
    ])

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_csv = self.input_dir / "train.csv"
        self.series_csv = self.input_dir / "train_series.csv"
        self.train_images_dir = self.input_dir / "train_images"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

cfg = Config()
seed_everything(cfg.seed)