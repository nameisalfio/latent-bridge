import torch

def get_device() -> torch.device:
    """Auto-detects hardware: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def log_hardware():
    device = get_device()
    print(f"Execution Device: {device}")
