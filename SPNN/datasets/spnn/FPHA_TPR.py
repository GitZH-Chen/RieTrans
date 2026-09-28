import os
import torch as th
from torch.utils import data

class DatasetSPD(data.Dataset):
    def __init__(self, path, names):
        self._path = path
        self._names = names

    def __len__(self):
        return len(self._names)

    def __getitem__(self, item):
        # Properly join the path and name of the file
        file_path = os.path.join(self._path, self._names[item])

        # Load the .pt file
        data = th.load(file_path)

        # Extract the covariance matrices (assuming they are stored with the key 'covariance_matrices')
        covariance_matrices = data['covariance_matrices']

        # Convert to PyTorch tensor and add the batch dimension (if needed)
        x = covariance_matrices.double()

        # Extract the label from the filename (assumed to be the last part after '_')
        y = int(self._names[item].split('.')[0].split('_')[-1])

        # Convert the label to a tensor
        y = th.tensor(y).long()

        return x, y


class DataLoaderFPHA_TPR:
    def __init__(self, data_path, batch_size):
        path_train = os.path.join(data_path, 'train')
        path_test = os.path.join(data_path, 'val')

        # Get the list of filenames in the train and test directories
        names_train = sorted([f for f in os.listdir(path_train) if f.endswith('.pt')])
        names_test = sorted([f for f in os.listdir(path_test) if f.endswith('.pt')])

        # Create DataLoader objects for train and test datasets
        self._train_generator = data.DataLoader(DatasetSPD(path_train, names_train), batch_size=batch_size, shuffle=True)
        self._test_generator = data.DataLoader(DatasetSPD(path_test, names_test), batch_size=batch_size, shuffle=False)

