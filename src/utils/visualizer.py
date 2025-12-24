import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List, Optional
import wandb

class Visualizer:
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        # Mean and Std for CIFAR-10 to reverse normalization
        self.mean = np.array([0.4914, 0.4822, 0.4465])
        self.std = np.array([0.2023, 0.1994, 0.2010])

    def _unnormalize(self, img: np.ndarray) -> np.ndarray:
        """Reverses the normalization for visualization."""
        if img.shape[0] == 3: # Handle CHW
            img = img.transpose((1, 2, 0)) 
        img = self.std * img + self.mean
        img = np.clip(img, 0, 1)
        return img

    def plot_predictions(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor, 
        preds: Optional[torch.Tensor] = None, 
        max_imgs: int = 8
    ):
        num_imgs = min(len(images), max_imgs)
        fig, axes = plt.subplots(1, num_imgs, figsize=(15, 3))
        if num_imgs == 1: axes = [axes]
        
        images = images.cpu().numpy()
        labels = labels.cpu().numpy()
        if preds is not None:
            preds = preds.cpu().numpy()

        for i in range(num_imgs):
            img = self._unnormalize(images[i])
            axes[i].imshow(img)
            gt_label = self.class_names[labels[i]]
            title = f"GT: {gt_label}"
            
            if preds is not None:
                pred_label = self.class_names[preds[i]]
                color = "green" if preds[i] == labels[i] else "red"
                axes[i].set_title(f"{title}\nPred: {pred_label}", color=color, fontsize=10)
            else:
                axes[i].set_title(title, fontsize=10)
            axes[i].axis("off")
        
        plt.tight_layout()
        return fig

    def plot_learning_curves(self, entity: str, project: str, run_ids: Optional[List[str]] = None):
        """
        Fetches metrics from WandB and plots Train/Val Loss and Val Accuracy.
        """
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}")
        
        all_data = []
        for run in runs:
            # If run_ids is provided, filter specifically for those
            if run_ids and run.id not in run_ids:
                continue
                
            history = run.history(keys=["epoch", "train/loss", "val/loss", "val/accuracy"])
            history["model_run"] = f"{run.name} ({run.id})"
            all_data.append(history)

        if not all_data:
            print("No data found for the specified project/runs.")
            return None

        df = pd.concat(all_data).reset_index(drop=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Loss Curves (Train vs Val)
        sns.lineplot(data=df, x="epoch", y="train/loss", hue="model_run", ax=ax1, linestyle="--", alpha=0.5)
        sns.lineplot(data=df, x="epoch", y="val/loss", hue="model_run", ax=ax1, linewidth=2)
        ax1.set_title("Loss: Training (--) vs Validation (-)")
        ax1.set_yscale("log")
        ax1.grid(True, which="both", ls="-", alpha=0.2)

        # 2. Accuracy Curve
        sns.lineplot(data=df, x="epoch", y="val/accuracy", hue="model_run", ax=ax2, linewidth=2)
        ax2.set_title("Validation Accuracy (%)")
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.2)

        plt.tight_layout()
        return fig

    def log_to_wandb(self, images: torch.Tensor, labels: torch.Tensor, preds: torch.Tensor, step: int):
        wandb_images = []
        for i in range(min(len(images), 10)):
            img = self._unnormalize(images[i].cpu().numpy())
            wandb_images.append(wandb.Image(
                img, 
                caption=f"GT: {self.class_names[labels[i]]}, Pred: {self.class_names[preds[i]]}"
            ))
        wandb.log({"visuals/predictions": wandb_images}, step=step)