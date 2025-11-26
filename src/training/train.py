"""Universal training script using Hydra configuration."""

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from hydra.utils import instantiate
import torch
from pathlib import Path

# Get absolute path to configs directory
CONFIG_PATH = Path(__file__).parent.parent.parent / "configs"


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="train_mountain_car")
def main(cfg: DictConfig) -> None:
    """Main training function.
    
    This function is completely generic - it works for any system by
    instantiating components from the Hydra configuration.
    """
    # Set random seed
    pl.seed_everything(cfg.seed)
    
    # Instantiate data module
    print("Instantiating data module...")
    datamodule = instantiate(cfg.data)
    datamodule.setup()
    
    # Instantiate model
    print("Instantiating model...")
    model = instantiate(cfg.model)
    
    # Instantiate Lightning module
    print("Instantiating Lightning module...")
    from src.training.classifier_module import ClassifierModule
    lightning_module = ClassifierModule(
        model=model,
        optimizer_config=cfg.optimizer,
        learning_rate=cfg.base_lr,
    )
    
    # Instantiate trainer
    print("Instantiating trainer...")
    trainer = instantiate(cfg.trainer)
    
    # Train
    print("Starting training...")
    trainer.fit(lightning_module, datamodule)
    
    print("Training complete!")


if __name__ == "__main__":
    main()

