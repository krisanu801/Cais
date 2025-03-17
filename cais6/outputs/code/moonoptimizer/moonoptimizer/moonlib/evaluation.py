import torch

def calculate_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculates the accuracy of the model.

    Args:
        predictions: The predicted labels.
        labels: The ground truth labels.

    Returns:
        The accuracy of the model.
    """
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / total