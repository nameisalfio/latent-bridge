import pytest
import json
from pathlib import Path

# Thresholds for CI/CD acceptance
MIN_ACCURACY_CNN = 0.65  # 65% minimum for SpatialCNN
MIN_ACCURACY_MLP = 0.40  # 40% minimum for PixelMLP

def load_metrics(model_name):
    """Utility to load metrics from the results directory."""
    metrics_path = Path(f"results/eval_benchmarks/{model_name}/metrics.json")
    if not metrics_path.exists():
        pytest.fail(f"Metrics file not found for {model_name}. Did you run scripts/test.py?")
    
    with open(metrics_path, 'r') as f:
        return json.load(f)

def test_cnn_performance():
    """Verify that SpatialCNN meets the minimum quality bar."""
    metrics = load_metrics("SpatialCNN")
    accuracy = metrics['summary']['accuracy']
    assert accuracy >= MIN_ACCURACY_CNN, f"SpatialCNN accuracy {accuracy} is below threshold {MIN_ACCURACY_CNN}"

def test_mlp_performance():
    """Verify that PixelMLP meets the minimum quality bar."""
    metrics = load_metrics("PixelMLP")
    accuracy = metrics['summary']['accuracy']
    assert accuracy >= MIN_ACCURACY_MLP, f"PixelMLP accuracy {accuracy} is below threshold {MIN_ACCURACY_MLP}"

def test_artifact_existence():
    """Ensure all required plots were generated during evaluation."""
    for model in ["SpatialCNN", "PixelMLP"]:
        base_path = Path(f"results/eval_benchmarks/{model}")
        assert (base_path / "confusion_matrix.png").exists(), f"Missing CM plot for {model}"
        assert (base_path / "roc_curves.png").exists(), f"Missing ROC plot for {model}"