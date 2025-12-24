from typing import List
import os
import torch
from onnx2torch import convert

def load_teacher(weights_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load a teacher model from either .pth or .onnx format and freeze it.
    
    Args:
        weights_path: Path to the model weights file
        device: Device to load the model on
        
    Returns:
        Loaded and frozen teacher model
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Teacher weights not found at: {weights_path}")

    if weights_path.endswith(".onnx"):
        model = convert(weights_path).to(device)
    elif weights_path.endswith(".pth"):
        # MyArcFace models are saved as entire model objects
        model = torch.load(weights_path, map_location=device).to(device)
    else:
        raise ValueError(f"Unsupported teacher weight format: {weights_path}")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def load_teacher_ensemble(weight_dir: str, teacher_files: List[str],
                          device: torch.device) -> List[torch.nn.Module]:
    """
    Load multiple teacher models to form an ensemble.
    
    Args:
        weight_dir: Directory containing the model weights
        teacher_files: List of model filenames
        device: Device to load the models on
        
    Returns:
        List of loaded and frozen teacher models
    """
    models = []
    for f in teacher_files:
        path = os.path.join(weight_dir, f)
        m = load_teacher(path, device)
        models.append(m)
    return models