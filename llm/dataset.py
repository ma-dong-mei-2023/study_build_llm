import torch
from torch.utils.data import Dataset, DataLoader


class GPTDataset(Dataset):
    """滑动窗口文本数据集。"""

    def __init__(self, token_ids, context_length, stride):
        self.inputs  = []
        self.targets = []
        for i in range(0, len(token_ids) - context_length, stride):
            self.inputs.append(torch.tensor(token_ids[i : i + context_length]))
            self.targets.append(torch.tensor(token_ids[i + 1 : i + context_length + 1]))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def make_dataloader(token_ids, context_length, stride, batch_size, shuffle=True):
    ds = GPTDataset(token_ids, context_length, stride)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)
