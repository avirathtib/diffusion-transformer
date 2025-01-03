import torch

def update_ema(ema_model, model, alpha=0.999):
    """Update the EMA model weights."""
    with torch.no_grad():
        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(model_param.data, alpha=1 - alpha)
