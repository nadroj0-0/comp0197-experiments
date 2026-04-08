from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def init_model(images, dropout_prob=0.0):
    from utils.network import SalesGRU

    print("\nCreating model...")
    model = SalesGRU(dropout_prob).to(device)
    print(model)
    print("\nTesting forward pass...")
    images = images.to(device)
    outputs = model(images)
    print("Model output shape:", outputs.shape)
    return model, outputs


def init_loss(model, train_loader, loss_fn=None):
    print("\nCreating loss function...")
    criterion = loss_fn if loss_fn is not None else nn.MSELoss()
    if loss_fn is not None:
        print("Custom loss function — skipping initial loss test")
        return criterion, None
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        preds = model(x)
    loss = criterion(preds, y)
    print(f"Initial loss: {loss.item():.4f}")
    return criterion, loss


def init_optimiser(model, method, **kwargs):
    import inspect

    print("\nCreating optimiser...")
    if not hasattr(optim, method):
        raise ValueError(f"Optimizer {method} not found in torch.optim")
    optimiser_class = getattr(optim, method)
    try:
        optim_method = optimiser_class(model.parameters(), **kwargs)
    except TypeError:
        expected_signature = inspect.signature(optimiser_class)
        print(f"\nInvalid arguments for optimizer '{method}'")
        print("Expected constructor signature:")
        print(f"{method}{expected_signature}")
        raise TypeError(
            f"Invalid arguments for optimizer '{method}'."
            f"Expected signature: {method}{expected_signature}"
        )
    print("Optimiser created:", optim_method)
    return optim_method

