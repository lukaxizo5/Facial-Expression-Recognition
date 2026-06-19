from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.constants import IMAGE_SIZE, FER2013_MEAN, FER2013_STD


class FER2013Dataset(Dataset):
    def __init__(self, csv_path, split, transform=None, normalize="standard"):
        self.csv_path = Path(csv_path)
        self.split = split
        self.transform = transform
        self.normalize = normalize

        df = pd.read_csv(self.csv_path)
        df.columns = df.columns.str.strip()

        self.df = df[df["Usage"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No rows found for split: {split}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        label = int(row["emotion"])

        pixels = np.fromstring(row["pixels"], sep=" ", dtype=np.float32)
        image = pixels.reshape(IMAGE_SIZE, IMAGE_SIZE)

        # Scale to [0, 1]
        image = image / 255.0

        if self.normalize == "standard":
            image = (image - FER2013_MEAN) / FER2013_STD
        elif self.normalize == "scale":
            pass
        elif self.normalize == "none":
            image = pixels.reshape(IMAGE_SIZE, IMAGE_SIZE)
        else:
            raise ValueError(f"Unknown normalize option: {self.normalize}")

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform is not None:
            image = self.transform(image)

        return image, label