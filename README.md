# Latent-Bridge: Probing Spatial Inductive Biases

**Latent-Bridge** is a comparative study between Fully Connected Networks (PixelMLP) and Convolutional Neural Networks (SpatialCNN) on the CIFAR-10 dataset. The project explores how different architectures bridge the gap between raw pixel intensities and semantic labels, specifically focusing on translation invariance and spatial robustness.

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/nameisalfio/latent-bridge.git
cd latent-bridge

# Install dependencies
pip install -r requirements.txt
```

### 2. Training

We use Hydra for configuration management. You can launch training for either architecture:

```bash
# Train Spatial CNN 
python scripts/train.py model=spatial_cnn

# Train Pixel MLP
python scripts/train.py model=pixel_mlp
```

### 3. Resuming from Checkpoint

If a training session was interrupted, resume using the `resume_from_checkpoint` flag:

```bash
# Example: Resuming PixelMLP
python scripts/train.py \
    model=pixel_mlp \
    trainer.resume_from_checkpoint=checkpoints/PixelMLP/model_epoch_5.pt
```

---

## 📊 Project Architecture

The project is designed with a modular MLOps approach:

- **`configs/`**: Modular YAML files (Hydra) for models, datasets, and trainers.
- **`src/`**: Core logic containing the `ClassificationEvaluator`, `Visualizer`, and model definitions.
- **`scripts/`**: Entry points for `train.py`, `test.py`, and `inference.py`.
- **`results/`**: Persistent storage for metrics, confusion matrices, and ROC curves.
- **`tests/`**: CI/CD suite for model regression testing.

---

## 🧪 Evaluation & Benchmarking

The evaluation suite goes beyond simple accuracy to provide a deep dive into model performance.

### Running the Test Suite

```bash
python scripts/test.py model=spatial_cnn
```

This generates a comprehensive report in `results/eval_benchmarks/`, including:

- **`metrics.json`**: Macro/Weighted F1, MCC, and AUC-ROC.
- **`confusion_matrix.png`**: Heatmap of classification errors.
- **`roc_curves.png`**: One-vs-Rest ROC analysis for all 10 classes.

### The Shift Test (Invariance Analysis)

Located in `notebooks/exploration.ipynb`, this test applies a spatial translation (shift) to images.

- **Hypothesis**: The PixelMLP accuracy will collapse due to lack of spatial priors.
- **Result**: The SpatialCNN maintains performance due to convolutional weight sharing.

---

## 🔍 Inference

Test the model on single external images:

```bash
python scripts/inference.py model=spatial_cnn image_path=path/to/your_image.jpg
```

---

## 🛠 CI/CD and Quality Assurance

We implement **Model Regression Testing**. Every push to the repository triggers a pipeline that verifies:

- **Code Integrity**: No breaking changes in the pipeline.
- **Performance Bar**: SpatialCNN must maintain >65% accuracy on the standardized test set.

Run tests locally:

```bash
pytest tests/test_model_performance.py
```

---

## 📈 Key Findings

| Architecture | Accuracy | F1 Macro | Spatial Robustness |
|--------------|----------|----------|--------------------|
| **SpatialCNN** | ~72%     | 0.71     | High               |
| **PixelMLP**   | ~45%     | 0.43     | Low                |

**Conclusion**: Convolutional layers act as a superior bridge for image data by enforcing local connectivity and translation invariance, whereas MLPs require significantly more data to learn the same spatial relationships.