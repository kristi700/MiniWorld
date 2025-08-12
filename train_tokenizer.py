import hydra

from utils.model_builder import build_model 
from data.data_builder import prepare_dataloaders
from utils.train_utils import get_transforms, setup_device

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config):
    """Main training function."""

    device = setup_device()
    transforms = get_transforms(config)

    train_loader, val_loader = prepare_dataloaders(config, transforms, config["training"]["type"].lower())
    model = build_model(config).to(device)
    
    # trainer = get_trainer(
    #     mode,
    #     model,
    #     get_save_path(config),
    #     config,
    #     train_loader,
    #     val_loader,
    #     device,
    #     start_epoch=start_epoch,
    #     best_val_loss=best_val_loss,
    # )
    # trainer.fit(config["training"]["num_epochs"])

if __name__ == "__main__":
    main()