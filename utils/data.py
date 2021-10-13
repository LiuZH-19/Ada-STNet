import os
from typing import Dict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class ZScoreScaler:
    def __init__(self, mean: float, std: float):
        assert std > 0
        self.mean = mean
        self.std = std

    def transform(self, x: Tensor, nan_val: float):
        zeros = torch.eq(x, nan_val)
        x = (x - self.mean) / self.std
        x[zeros] = 0.0
        return x

    def inverse_transform(self, x: Tensor, nan_val: float):
        zeros = torch.eq(x, nan_val)
        x = x * self.std + self.mean
        x[zeros] = 0.0
        return x


class TrafficPredictionDataset(Dataset):
    def __init__(self, data_path: str, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        data = np.load(data_path)

        # [num_samples, seq_length, num_nodes, num_features]
        self.inputs: np.ndarray = data['x']
        self.targets: np.ndarray = data['y']
        assert self.inputs.shape[0] == self.targets.shape[0]
        assert self.inputs.shape[2] == self.targets.shape[2]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        x, y = self.inputs[idx][..., :self.input_dim], self.targets[idx][..., :self.output_dim]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    @property
    def std(self) -> float:
        data = np.concatenate([self.inputs[..., :1], self.targets[..., :1]], 1)
        return data[np.not_equal(data, 0.0)].std()

    @property
    def mean(self) -> float:
        data = np.concatenate([self.inputs[..., :1], self.targets[..., :1]], 1)
        return data[np.not_equal(data, 0.0)].mean()


def get_datasets(dataset: str, input_dim: int, output_dim: int) -> Dict[str, TrafficPredictionDataset]:
    return {key: TrafficPredictionDataset(os.path.join('data', dataset, f'{key}.npz'), input_dim, output_dim) for key in
            ['train', 'val', 'test']}


def get_dataloaders(datasets: Dict[str, Dataset],
                    batch_size: int,
                    num_workers: int = 16) -> Dict[str, DataLoader]:
    return {key: DataLoader(dataset=ds,
                            batch_size=batch_size,
                            shuffle=(key == 'train'),
                            num_workers=num_workers) for key, ds in datasets.items()}
