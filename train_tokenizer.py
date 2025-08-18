import os
import hydra
import shutil

from utils.model_builder import build_model 
from data.data_builder import prepare_dataloaders
from utils.train_utils import get_transforms, setup_device
from utils.trainers.tokenizer_trainer import TokenizerTrainer

def get_trainer(
    mode,
    model,
    save_path,
    config,
    train_loader,
    val_loader,
    device,
):
    """Get the appropriate trainer based on training mode."""
    if mode == 'tokenizer':
        trainer = TokenizerTrainer(
            model, save_path, config, train_loader, val_loader, device
        )  
    else:
        raise ValueError(f"Unknown training mode: {mode}")

    return trainer

def get_save_path(config):
    """Get the save path from Hydra configuration."""
    resume_model = config['training'].get('resume_from_checkpoint', None)
    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if resume_model is not None:
        resume_path=os.path.dirname(resume_model)
        assert os.path.exists(resume_path), f"resume_from_checkpoint: {resume_path} does not exist!"
        shutil.rmtree(hydra_output_dir, ignore_errors=True)

        return resume_path
    else:
        return hydra_output_dir

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config):
    """Main training function."""

    device = setup_device()
    transforms = get_transforms(config)
    mode = config["training"]["type"].lower()
    
    train_loader, val_loader = prepare_dataloaders(config, transforms, config["training"]["type"].lower())
    model = build_model(config).to(device)
    trainer = get_trainer(
        mode,
        model,
        get_save_path(config),
        config,
        train_loader,
        val_loader,
        device,
    )
    trainer.fit(config["training"]["num_epochs"])

if __name__ == "__main__":
    main()