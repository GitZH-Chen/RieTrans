"""Radar dataset loader for the fixed-encoder SPDNN experiments."""

import os
import random

import numpy as np
import torch
from torch.utils import data


class DatasetRadar(data.Dataset):
    def __init__(self, path, names):
        self.path = path
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, item):
        name = self.names[item]
        signal = np.load(os.path.join(self.path, name))
        signal = np.stack((signal.real, signal.imag), axis=0)
        label = int(name.rsplit("_", 1)[-1].split(".")[0])
        return torch.from_numpy(signal), torch.tensor(label).long()


class DataLoaderRadar:
    def __init__(self, data_path, pval, ptest, batch_size, seed):
        names = sorted(
            name for name in os.listdir(data_path) if name.endswith(".npy")
        )
        random.Random(seed).shuffle(names)

        validation_size = int(pval * len(names))
        test_size = int(ptest * len(names))
        train_start = validation_size + test_size

        test_set = DatasetRadar(data_path, names[:test_size])
        validation_set = DatasetRadar(
            data_path,
            names[test_size:train_start],
        )
        train_set = DatasetRadar(data_path, names[train_start:])

        generator = torch.Generator().manual_seed(seed)
        self._train_generator = data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        self._test_generator = data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
        )
        self._val_generator = data.DataLoader(
            validation_set,
            batch_size=batch_size,
            shuffle=False,
        )
