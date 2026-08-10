import os
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import cv2
import pydicom
import torch
from torch.utils.data import Dataset
from config import cfg

class RSNAKneeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, is_train: bool = True, transform=None):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def _read_dicom_slice(self, path: Path) -> np.ndarray:
        try:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array.astype(np.float32)
            
            window_center = getattr(dcm, "WindowCenter", None)
            window_width = getattr(dcm, "WindowWidth", None)

            if isinstance(window_center, pydicom.multival.MultiValue):
                window_center = window_center[0]
            if isinstance(window_width, pydicom.multival.MultiValue):
                window_width = window_width[0]

            if window_center is not None and window_width is not None:
                img_min = window_center - (window_width / 2.0)
                img_max = window_center + (window_width / 2.0)
                img = np.clip(img, img_min, img_max)

            img_min, img_max = img.min(), img.max()
            if img_max - img_min > 0:
                img = (img - img_min) / (img_max - img_min)
            else:
                img = np.zeros_like(img)

            return (img * 255.0).astype(np.uint8)
        except Exception:
            return np.zeros((cfg.image_size, cfg.image_size), dtype=np.uint8)

    def _get_volume_stack(self, series_path: Path) -> np.ndarray:
        if not series_path.exists():
            return np.zeros((cfg.image_size, cfg.image_size, cfg.num_slices), dtype=np.uint8)

        slice_files = sorted(
            list(series_path.glob("*.dcm")),
            key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem
        )
        
        total_slices = len(slice_files)
        if total_slices == 0:
            return np.zeros((cfg.image_size, cfg.image_size, cfg.num_slices), dtype=np.uint8)

        indices = np.linspace(0, total_slices - 1, cfg.num_slices, dtype=int)
        stacked_slices = []

        for idx in indices:
            img = self._read_dicom_slice(slice_files[idx])
            img = cv2.resize(img, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_AREA)
            stacked_slices.append(img)

        return np.stack(stacked_slices, axis=-1)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        study_id = str(row.get("study_id", row.get("StudyInstanceUID", "")))
        series_id = str(row.get("series_id", ""))
        
        series_path = cfg.train_images_dir / study_id / series_id
        volume = self._get_volume_stack(series_path)

        if self.transform is not None:
            augmented = self.transform(image=volume)
            volume = augmented["image"]

        volume = volume.astype(np.float32) / 255.0
        volume = np.transpose(volume, (2, 0, 1))
        volume = np.stack([volume] * 3, axis=0)

        data_dict = {
            "image": torch.tensor(volume, dtype=torch.float32)
        }

        if self.is_train:
            available_cols = [col for col in cfg.target_columns if col in row.index]
            if len(available_cols) == len(cfg.target_columns):
                labels = row[cfg.target_columns].values.astype(np.float32)
            else:
                labels = np.zeros(cfg.num_classes, dtype=np.float32)
            
            data_dict["label"] = torch.tensor(labels, dtype=torch.float32)

        return data_dict