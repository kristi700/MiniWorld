from torch.utils.data import Dataset

class MiniWorldDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        ...

    def __len__(self):
        return 1 # len(self.data)

    def __getitem__(self, idx):
        ...