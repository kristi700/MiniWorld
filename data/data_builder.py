import torch

from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split, Subset

from .datasets import MiniWorldDataset

def _get_dataset(config, mode, transforms):
    """
    Internal helper to instantiate the correct dataset.
    """
    config_section = config.get("eval", {}) if "eval" in mode else config.get("data", {})

    dataset_name = config_section.get(
        "dataset_name", config.get("data", {}).get("dataset_name", "")
    ).lower()
    data_dir = config_section.get(
        "data_dir", config.get("data", {}).get("data_dir")
    )

    if mode in ["tokenizer"]:
        if dataset_name.lower() == 'miniworld':
            return MiniWorldDataset(
                data_dir, transform=transforms["train"]
            ), MiniWorldDataset(data_dir, transform=transforms["val"])
        else:
           raise ValueError(f"Unknown dataset name: {dataset_name}")
    else:
        raise ValueError(f"Unknown mode for dataset creation: {mode}")


def prepare_dataloaders(config, transforms, mode):
    """
    Prepare train and validation dataloaders based on the configuration.

    This function is the single entry point for creating dataloaders for any
    training or evaluation mode.

    Args:
        config (OmegaConf): The global configuration object.
        transforms (dict): A dictionary of transforms, e.g., {'train': ..., 'val': ...}.
        mode(str): type of dataset

    Returns:
        tuple: (train_loader, val_loader)
    """
    if OmegaConf.is_list(mode):
        data_loading_mode = mode[0]
    else:
        data_loading_mode = mode.lower()

    train_dataset_full, val_dataset_full = _get_dataset(
        config, data_loading_mode, transforms
    )

    total_size = len(train_dataset_full)
    val_split_ratio = config.data.val_split

    if val_split_ratio <= 0 or val_split_ratio >= 1:
        train_size = total_size
        val_size = 0
    else:
        val_size = int(total_size * val_split_ratio)
        train_size = total_size - val_size

    generator = torch.Generator().manual_seed(config.training.random_seed)

    if val_size > 0:
        train_subset_indices, val_subset_indices = random_split(
            range(total_size), [train_size, val_size], generator=generator
        )
        train_dataset = Subset(train_dataset_full, train_subset_indices.indices)
        val_dataset = Subset(val_dataset_full, val_subset_indices.indices)
    else:

        train_dataset = train_dataset_full
        val_dataset = None

    batch_size = config.get("training", {}).get(
        "batch_size", config.get("eval", {}).get("batch_size")
    )
    num_workers = config.data.num_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

    return train_loader, val_loader
