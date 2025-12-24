import os
import hydra
import wandb
import torch
import torch.nn as nn
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from src.utils.device import get_device, log_hardware
from src.data.datamodule import get_dataloader 

@hydra.main(version_base=None, config_path="../configs", config_name="main")
def train(cfg: DictConfig):
    log_hardware()
    device = get_device()

    # 1. Initialize Experiment Tracker and Checkpoint Directory
    if cfg.trainer.use_wandb:
        wandb.init(
            project="latent-bridge",
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"{cfg.model._target_.split('.')[-1]}_{cfg.dataset.name}"
        )
    model_name = cfg.model._target_.split(".")[-1]
    ckpt_dir = Path(cfg.trainer.checkpoint_dir) / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 2. Setup Dataloader & Model
    train_loader = get_dataloader(cfg, train=True)
    model: nn.Module = hydra.utils.instantiate(cfg.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = hydra.utils.instantiate(cfg.trainer.optimizer, params=model.parameters())

    # 3. Training Loop
    model.train()
    for epoch in range(cfg.trainer.max_epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad() 
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        
        # 4. Metrics & Visualization
        metrics = {
            "epoch": epoch,
            "train/loss": avg_loss,
            "lr": optimizer.param_groups[0]['lr']
        }
        
        if cfg.trainer.use_wandb:
            wandb.log(metrics)
        print(f"Epoch {epoch}: {metrics}")

        # 5. Structured Checkpointing
        if (epoch) % cfg.trainer.checkpoint_interval == 0:
            checkpoint_path = ckpt_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)

    # Save final checkpoint
    final_path = ckpt_dir / "last_checkpoint.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, final_path)
    print(f"\nLast checkpoint saved to: {final_path}")

    if cfg.trainer.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    train()