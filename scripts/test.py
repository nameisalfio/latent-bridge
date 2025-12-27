import torch
import hydra
import logging
import numpy as np
from pathlib import Path
from src.data.datamodule import get_dataloader
from src.utils.device import get_device
from src.utils.visualizer import plot_confusion_matrix, plot_roc_curves
from src.evaluation.evaluator import ClassificationEvaluator

logger = logging.getLogger("latent-bridge-eval")

@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(cfg):
    device = get_device()
    model_name = cfg.model._target_.split(".")[-1]
    
    # Define a clean results directory at the project root
    root_dir = Path(hydra.utils.get_original_cwd())
    results_dir = root_dir / "results" / "eval_benchmarks" / model_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Model and Data Setup
    test_loader = get_dataloader(cfg, train=False)
    model = hydra.utils.instantiate(cfg.model).to(device)
    
    checkpoint_path = Path(cfg.trainer.checkpoint_dir) / model_name / "model_final.pt"
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Comprehensive Inference (Collecting Predictions and Probabilities)
    y_true, y_pred, y_probs = [], [], []
    logger.info(f"Running full evaluation for {model_name}...")
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            probs = torch.softmax(outputs, dim=1)
            
            y_true.extend(labels.numpy())
            y_pred.extend(outputs.argmax(dim=1).cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    y_true, y_pred, y_probs = np.array(y_true), np.array(y_pred), np.array(y_probs)

    # 3. Metrics Computation
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    evaluator = ClassificationEvaluator(class_names)
    metrics = evaluator.compute_all_metrics(y_true, y_pred, y_probs)

    # 4. Artifact Archiving
    evaluator.save_metrics(metrics, results_dir / "metrics.json")
    plot_confusion_matrix(np.array(metrics['confusion_matrix']), class_names, results_dir / "confusion_matrix.png")
    plot_roc_curves(y_true, y_probs, class_names, results_dir / "roc_curves.png")

    logger.info(f"Evaluation finalized. Results archived in: {results_dir}")

if __name__ == "__main__":
    main()