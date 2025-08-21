import torch

def top_k_accuracy(output, target, k=1):
    """Compute the top-k accuracy, but ensure k ≤ number of classes"""
    with torch.no_grad():
        max_k = min(k, output.size(1))  # Prevent k > number of classes
        _, pred = output.topk(max_k, dim=1, largest=True, sorted=True)
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        return correct[:, :max_k].reshape(-1).float().sum(0).item()

