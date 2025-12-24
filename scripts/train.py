import logging
import random
import numpy as np
import torch
import wandb
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from collections import OrderedDict
from src.utils.device import get_device, log_hardware
from src.data.datamodule import get_dataloader 
from src.utils.visualizer import Visualizer

logger = logging.getLogger("latent-bridge")

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def do_save(model, optimizer, epoch, metrics, ckpt_dir, name):
    checkpoint_path = ckpt_dir / f"{name}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'wandb_run_id': wandb.run.id, 
        'metrics': metrics,
    }, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")

@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(cfg: DictConfig):
    # --- 1. ENVIRONMENT AND REPRODUCIBILITY ---
    set_seed(cfg.trainer.seed)
    device = get_device()
    log_hardware()

    # --- 2. IDENTITY AND PATH MANAGEMENT ---
    model_name = cfg.model._target_.split(".")[-1]
    ckpt_dir = Path(cfg.trainer.checkpoint_dir) / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 3. COMPONENT INSTANTIATION ---
    train_loader = get_dataloader(cfg, train=True)
    val_loader = get_dataloader(cfg, train=False)
    
    model = hydra.utils.instantiate(cfg.model).to(device)
    optimizer = hydra.utils.instantiate(cfg.trainer.optimizer, params=model.parameters())
    criterion = hydra.utils.instantiate(cfg.trainer.criterion)
    
    # --- 4. RESUME LOGIC ---
    start_epoch = 0
    recovered_run_id = None
    if cfg.trainer.resume_from_checkpoint:
        ckpt_path = Path(cfg.trainer.resume_from_checkpoint)
        if ckpt_path.exists():
            logger.info(f"Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            recovered_run_id = ckpt.get('wandb_run_id')
        else:
            logger.warning(f"Checkpoint not found at {ckpt_path}. Starting from scratch.")

    # --- 5. WANDB INITIALIZATION ---
    run_id = recovered_run_id or cfg.trainer.get("wandb_run_id")
    if not cfg.trainer.get("no_wandb", False):
        run_name = cfg.trainer.get("wandb_run_name") or f"{model_name}_{cfg.dataset.name}"
        wandb.init(
            project=cfg.logging.project_name,
            id=run_id,         
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=str(ckpt_dir), 
            resume="allow"     
        )

    # --- 6. VISUALIZATION PREPARATION ---
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    viz = Visualizer(classes)

    logger.info(f"Starting training from epoch {start_epoch}")
    
    # --- 7. TRAINING LOOP ---
    for epoch in range(start_epoch, cfg.trainer.max_epochs):
        model.train()
        storage = OrderedDict() 
        train_loss, train_correct, train_total = 0.0, 0, 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (output.argmax(1) == target).sum().item()
            train_total += target.size(0)

        storage["epoch"] = epoch
        storage["train_loss"] = train_loss / len(train_loader)
        storage["train_acc"] = 100. * train_correct / train_total

        # --- 8. EVALUATION ---
        if (epoch + 1) % cfg.trainer.eval_period == 0:
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for v_data, v_target in val_loader:
                    v_data, v_target = v_data.to(device), v_target.to(device)
                    v_output = model(v_data)
                    val_loss += criterion(v_output, v_target).item()
                    val_correct += (v_output.argmax(1) == v_target).sum().item()
                    val_total += v_target.size(0)
            
            storage["val_loss"] = val_loss / len(val_loader)
            storage["val_acc"] = 100. * val_correct / val_total
            storage["lr"] = optimizer.param_groups[0]["lr"]

            if wandb.run:
                wandb.log(dict(storage), step=epoch)
                s_data, s_target = next(iter(val_loader))
                s_preds = model(s_data.to(device)).argmax(1)
                viz.log_to_wandb(s_data, s_target, s_preds, step=epoch)

            log_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in storage.items()])
            logger.info(log_str)

        # --- 9. PERIODIC CHECKPOINTING ---
        if (epoch + 1) % cfg.trainer.checkpoint_period == 0:
            do_save(model, optimizer, epoch, dict(storage), ckpt_dir, f"model_epoch_{epoch}")

    # Final Save
    do_save(model, optimizer, epoch, dict(storage), ckpt_dir, "model_final")
    
    if wandb.run: 
        wandb.finish()

if __name__ == "__main__":
    main()