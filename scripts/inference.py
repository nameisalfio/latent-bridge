import torch
import hydra
import logging
from pathlib import Path
from PIL import Image
from torchvision import transforms
from omegaconf import DictConfig
from src.utils.device import get_device

# Standard logger setup
logger = logging.getLogger("latent-bridge-inference")

def get_preprocessing():
    """
    Standard CIFAR-10 preprocessing used during training.
    """
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], 
            std=[0.2023, 0.1994, 0.2010]
        )
    ])

@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(cfg: DictConfig):
    # 1. Setup Device
    device = get_device()
    
    # 2. Path Handling
    img_path = cfg.paths.image_path
    if img_path is None:
        logger.error("Please provide an image path: python scripts/inference.py image_path=path/to/img.jpg")
        return
    
    model_name = cfg.model._target_.split(".")[-1]
    checkpoint_path = Path(cfg.trainer.checkpoint_dir) / model_name / "model_final.pt"

    # 3. Model Restoration
    model = hydra.utils.instantiate(cfg.model).to(device)
    if checkpoint_path.exists():
        logger.info(f"Loading weights from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
    else:
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        return

    # 4. Image Loading and Preprocessing
    try:
        image = Image.open(img_path).convert('RGB')
        preprocess = get_preprocessing()
        input_tensor = preprocess(image).unsqueeze(0).to(device) # Add batch dimension
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return

    # 5. Execution
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)

    # 6. Output Results
    predicted_label = class_names[predicted_idx]
    logger.info("-" * 30)
    logger.info(f"IMAGE: {img_path}")
    logger.info(f"PREDICTION: {predicted_label.upper()}")
    logger.info(f"CONFIDENCE: {confidence.item():.2%}")
    logger.info("-" * 30)

if __name__ == "__main__":
    main()