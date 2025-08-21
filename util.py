import torch

def top_k_accuracy(output, target, k=1):
    """Compute the top-k accuracy"""
    with torch.no_grad():
        _, pred = output.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        return correct[:, :k].reshape(-1).float().sum(0).item()
