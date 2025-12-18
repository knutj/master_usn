import torch
import numpy as np
class ReadmissionDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, lengths):
        self.X = X  # shape: (N, T, F)
        self.y = y  # long for classification
        self.lengths = lengths

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_seq = torch.tensor(self.X[idx], dtype=torch.float32)
        y_val = torch.tensor(self.y[idx], dtype=torch.long)
        length = torch.tensor(self.lengths[idx], dtype=torch.long)
        return x_seq, length, x_seq, y_val  # return twice for LSTM and CNN input