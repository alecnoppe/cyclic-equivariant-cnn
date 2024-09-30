from torch.utils.data import Dataset
import torch


class MNIST_Dataset(Dataset):
    def __init__(self, path, transform=None, target_transform=None,):
        """
        Dataset object to read the MNIST dataset from a .pt file
        
        NOTE: First run make_datasets.py to generate the .pt files.

        Args:
            path: str; path to the .pt file
            transform: callable; transformation to apply to the input data
            target_transform: callable; transformation to apply to the target data
        """
        super().__init__()
        self.X, self.Y = torch.load(path)

    def __len__(self):
        """
        Returns the number of samples in the dataset
        """
        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        Returns the sample at the given index

        Args:
            idx: int; index of the sample

        Returns:
            torch.Tensor; input data
            torch.Tensor; target data
        """
        return self.X[idx,...], self.Y[idx,...]
