import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

class Visualizer:
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        # CIFAR-10 means and stds to reverse normalization
        self.mean = np.array([0.4914, 0.4822, 0.4465])
        self.std = np.array([0.2023, 0.1994, 0.2010])

    def _unnormalize(self, img: np.ndarray) -> np.ndarray:
        """Reverses normalization for visualization."""
        if img.shape[0] == 3: # If CHW, convert to HWC
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
        """Displays a grid of images with ground truth and predictions."""
        num_imgs = min(len(images), max_imgs)
        fig, axes = plt.subplots(1, num_imgs, figsize=(18, 4))
        if num_imgs == 1: axes = [axes]
        
        for i in range(num_imgs):
            img = self._unnormalize(images[i].cpu().numpy())
            axes[i].imshow(img)
            
            gt_label = self.class_names[labels[i]]
            title = f"GT: {gt_label}"
            
            if preds is not None:
                pred_label = self.class_names[preds[i]]
                color = "green" if preds[i] == labels[i] else "red"
                axes[i].set_title(f"{title}\nPRED: {pred_label}", color=color, fontsize=10)
            else:
                axes[i].set_title(title, fontsize=10)
                
            axes[i].axis("off")
        
        plt.tight_layout()
        return fig