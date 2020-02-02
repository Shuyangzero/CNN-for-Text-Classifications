from torch.utils.data import Dataset
import torch


class TextDataset(Dataset):
    def __init__(self, X, Y, stoi):
        self.X = X
        self.Y = torch.tensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
