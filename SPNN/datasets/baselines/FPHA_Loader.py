import os

import numpy as np
import torch as th

from torch.utils import data

class DatasetSPD(data.Dataset):
    def __init__(self, path, names):
        self._path = path
        self._names = names

    def __len__(self):
        return len(self._names)

    def __getitem__(self, item):
        file_name = os.path.join(self._path, self._names[item])
        x = np.load(file_name)[None, :, :]
        x = th.from_numpy(x)
        y = int(self._names[item].split('.')[0].split('_')[-1])
        y = th.from_numpy(np.array(y)).long()
        return x, y


class DataLoaderFPHA:
    def __init__(self, data_path, batch_size):
        path_train = os.path.join(data_path,'train')
        path_test = os.path.join(data_path,'val')
        for filenames in os.walk(path_train):
            names_train = sorted(filenames[2])
        for filenames in os.walk(path_test):
            names_test = sorted(filenames[2])
        self._train_generator = data.DataLoader(DatasetSPD(path_train, names_train), batch_size=batch_size,
                                                shuffle='True')
        self._test_generator = data.DataLoader(DatasetSPD(path_test, names_test), batch_size=batch_size,
                                               shuffle='False')
