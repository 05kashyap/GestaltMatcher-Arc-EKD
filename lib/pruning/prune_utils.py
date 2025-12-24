import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import List

def apply_unstructured_prune(model: nn.Module, amount: float, whitelist=(nn.Conv2d, nn.Linear)):
    """
    Apply magnitude-based unstructured pruning to specified layers and make it permanent.
    """
    for module in model.modules():
        if isinstance(module, whitelist) and hasattr(module, "weight"):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight") # Makes pruning permanent

def prune_iterative(model: nn.Module, schedule: List[float]):
    """
    Apply iterative pruning based on a schedule of pruning amounts.
    """
    for amt in schedule:
        apply_unstructured_prune(model, amount=amt)
    return model