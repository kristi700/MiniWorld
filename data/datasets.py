import os
import json
import bisect
import numpy as np

from PIL import Image
from functools import lru_cache
from torch.utils.data import Dataset

class MiniWorldDataset(Dataset):
    """
    Implements a memory efficient Dataset for handling larger number of datasamples using
    memmap objects and lru_cache for optimal speed/memory trade off.
    """
    def __init__(self, root_dir, transform=None):
        self.data_dir = root_dir
        self.transform = transform
        with open(os.path.join(root_dir, "index.json"), 'r') as file:
            self.index = json.load(file)

        self.paths = []
        self.cum = []
        self.total = 0
        for shard in self.index['shards']:
            self.paths.append(shard['path'])
            self.cum.append(shard['length'] if len(self.cum) == 0 else self.cum[-1] + shard['length'])
            self.total += shard['length']

        assert self.total == self.index['total'], "index.json total != Σ shards length, double check the data!"
            
    def __len__(self):
        """
        Returns the total number of frames in the dataset.
        """
        return self.total
    
    def __getitem__(self, idx):
        """
        Basic Dataset __getitem__ using, lru_cache for memmap objects to minimize I/O overhead for
        larger number of frames_shards.
        """
        if idx < 0 or idx >= self.total:
            raise IndexError
        
        shard = bisect.bisect_right(self.cum, idx)
        prev_end = 0 if shard == 0 else self.cum[shard - 1] 
        local_idx = idx - prev_end
        whole_data = open_memmap(os.path.join(self.data_dir, self.paths[prev_end]))
        frame = np.asarray(whole_data[local_idx])
        frame = Image.fromarray(frame) 
        if self.transform:
            frame = self.transform(frame)
        return frame

@lru_cache(maxsize=32)
def open_memmap(path: str) -> np.memmap:
    """
    Loads frames_shards as memmap objects for optimal memory consumption.
    """
    return np.load(path, mmap_mode="r")