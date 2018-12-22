import torch.utils.data as data


class IndexFeeder(data.Dataset):
    def __init__(self, roidb):
        self.roidb = roidb

    def __len__(self):
        return len(self.roidb)

    def __getitem__(self, index):
        return index
