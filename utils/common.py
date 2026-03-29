from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time
import json
from .early_stopping import EarlyStopping
from .data import *
from .training_strategies import *
from .network import *



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def init_model(images, dropout_prob=0.0):
    # Create the model
    print('\nCreating model...')
    model = SalesGRU(dropout_prob).to(device)
    print(model)
    # Test a forward pass
    print('\nTesting forward pass...')
    images = images.to(device)
    outputs = model(images)
    print('Model output shape:', outputs.shape)
    return model, outputs

def init_loss(model, train_loader, loss_fn=None):
    """
    Initialise any loss function and test on one batch.

    Args:
        model        : Instantiated model (already on device).
        train_loader : DataLoader to pull one test batch from.
        loss_fn      : Any nn.Module loss. Defaults to nn.MSELoss().
                       Pass any loss e.g. nn.CrossEntropyLoss(),
                       nn.BCEWithLogitsLoss(), or a custom NLL loss.

    Returns:
        criterion : The loss function.
        loss      : Initial loss value on one batch (for sanity checking).
    """
    print('\nCreating loss function...')
    criterion = loss_fn if loss_fn is not None else nn.MSELoss()
    if loss_fn is not None:
        print('Custom loss function — skipping initial loss test')
        return criterion, None
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        preds = model(x)
    loss = criterion(preds, y)
    print(f'Initial loss: {loss.item():.4f}')
    return criterion, loss


def init_optimiser(model, method, **kwargs):
    import inspect
    # Optimiser
    print('\nCreating optimiser...')
    # optimMethod = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    if not hasattr(optim, method):
        raise ValueError(f'Optimizer {method} not found in torch.optim')
    optimiser_class = getattr(optim, method)
    try:
        optim_method = optimiser_class(model.parameters(), **kwargs)
    except TypeError as e:
        expected_signature = inspect.signature(optimiser_class)
        print(f"\nInvalid arguments for optimizer '{method}'")
        print('Expected constructor signature:')
        print(f'{method}{expected_signature}')
        raise TypeError(
            f"Invalid arguments for optimizer '{method}'."
            f'Expected signature: {method}{expected_signature}'
        )
    #optim_method = optimiser_class(model.parameters(), lr=learn_rate, momentum=momentum_par)
    print('Optimiser created:', optim_method)
    return optim_method


def evaluate_model(data_loader, model, criterion, training_step=None, **kwargs):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if training_step is not None:
                loss, outputs = training_step(model, inputs, labels, criterion, **kwargs)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            # outputs = model(inputs)
            # loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            all_preds.append(outputs.detach().cpu())
            all_targets.append(labels.detach().cpu())
    average_loss = total_loss / total_samples
    accuracy = None
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    return average_loss, accuracy, all_preds, all_targets

def train_model(epochs, train_loader, val_loader, model, criterion, optim_method,
                training_step=gru_step, early_stopping_patience=None, early_stopping_min_delta=0.0,
                scheduler=None, clip_grad_norm=None, extra_metrics=None, **kwargs):
    # Training
    print('\nStarting training...')
    # num_epochs = 50
    history: dict = {'epoch_metrics': [],'early_stopping': None,'batch_losses': []}
    #batch_losses = []
    #epoch_losses = []
    accuracy_valid = getattr(training_step, "valid_train_accuracy", True)
    early_stopping_enabled = (
            early_stopping_patience is not None and early_stopping_patience > 0
    )
    early_stopper = EarlyStopping(early_stopping_patience,min_delta=early_stopping_min_delta) if early_stopping_enabled else None
    use_amp = device.type == "cuda" and not kwargs.get("disable_amp", False)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    for epoch in range(epochs):
        model.train()
        #epoch_loss = 0
        #num_batches = 0
        epoch_train_loss_sum = torch.tensor(0.0, device=device)
        epoch_train_correct = 0
        epoch_train_samples = 0

        for i, batch in enumerate(train_loader):
            # support both (x, y) and (x, y, weight) batches
            # weight is used by wquantile_gru_step for weighted pinball loss
            if len(batch) == 3:
                inputs, labels, batch_weight = batch
                kwargs["item_weights"] = batch_weight.to(device, non_blocking=True)
            else:
                inputs, labels = batch
                kwargs.pop("item_weights", None)
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optim_method.zero_grad()
            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    loss, outputs = training_step(model, inputs, labels, criterion, **kwargs)
                scaler.scale(loss).backward()
                if clip_grad_norm is not None:
                    scaler.unscale_(optim_method)
                    nn.utils.clip_grad_norm_(model.parameters(),max_norm=clip_grad_norm)
                scaler.step(optim_method)
                scaler.update()
            else:
                loss, outputs = training_step(model, inputs, labels, criterion, **kwargs)
                loss.backward()
                if clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(),max_norm=clip_grad_norm)
                optim_method.step()
            # #outputs = model(inputs)
            # #loss = criterion(outputs, labels)
            # loss, outputs = training_step(model,inputs,labels,criterion,**kwargs)
            # loss.backward()
            # if clip_grad_norm is not None:
            #     nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            # optim_method.step()
            batch_size = labels.size(0)
            # loss_value = loss.item()
            loss_value = loss.detach()
            history['batch_losses'].append({
                'epoch': epoch+1,
                'batch': i+1,
                # 'loss': loss_value
                'loss': loss_value.item()
            })
            #epoch_loss += loss_value
            #num_batches += 1
            epoch_train_loss_sum += loss_value * batch_size
            if accuracy_valid:
                predictions = outputs.argmax(dim=1)
                epoch_train_correct += (predictions == labels).sum().item()
            epoch_train_samples += batch_size
        #avg_epoch_loss = epoch_loss / num_batches
        #epoch_losses.append(avg_epoch_loss)
        # train_loss = epoch_train_loss_sum / epoch_train_samples
        train_loss = (epoch_train_loss_sum / epoch_train_samples).item()
        if accuracy_valid:
            train_accuracy = epoch_train_correct / epoch_train_samples
        eval_kwargs = {k: v for k, v in kwargs.items() if k != "item_weights"}
        val_loss, val_accuracy, val_preds, val_targets = evaluate_model(val_loader, model, criterion, training_step=training_step, **eval_kwargs)
        if scheduler is not None:
            scheduler.step(val_loss)
        epoch_record = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'validation_loss': val_loss,
            'validation_accuracy': val_accuracy
        }
        if accuracy_valid:
            epoch_record['train_accuracy'] = train_accuracy
        if extra_metrics is not None:
            for name, fn in extra_metrics.items():
                epoch_record[name] = fn(val_preds, val_targets)
        history['epoch_metrics'].append(epoch_record)
        if early_stopper:
            stop = early_stopper.update(val_loss, model, epoch + 1)
            if stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                early_stopper.triggered = True
                break
        if accuracy_valid:
            print(
                f"Epoch {epoch + 1:02d} | "
                f"train_loss={train_loss:.4f} | "
                f"train_acc={train_accuracy:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_accuracy:.4f}"
            )
        else:
            extra_str = ""
            if extra_metrics is not None:
                extra_str = " | " + " | ".join(
                    f"{k}={epoch_record[k]:.4f}" for k in extra_metrics
                )
            print(
                f"Epoch {epoch + 1:02d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f}"
                + extra_str
            )
        #print(f'Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}')
    print('Training finished.')
    if early_stopper and early_stopper.stopped_epoch is None:
        early_stopper.stopped_epoch = epochs
    best_val_accuracy = None
    if early_stopper and early_stopper.best_model_state is not None:
        model.load_state_dict(early_stopper.best_model_state)
        print("Restored best model from early stopping.")
        print(f"Best validation loss {early_stopper.best_val_loss:.4f} at epoch {early_stopper.best_epoch}")
        for m in history["epoch_metrics"]:
            if m["epoch"] == early_stopper.best_epoch:
                best_val_accuracy = m["validation_accuracy"]
                break
    if early_stopping_enabled:
        history["early_stopping"] = {
            "enabled": True,
            "triggered": early_stopper.triggered,
            "patience": early_stopper.patience,
            "min_delta": early_stopper.min_delta,
            "best_epoch": early_stopper.best_epoch,
            "best_validation_loss": early_stopper.best_val_loss,
            "best_validation_accuracy": best_val_accuracy,
            "stopped_epoch": early_stopper.stopped_epoch
        }
    #return batch_losses, epoch_losses
    return history

def save_model(model, name, model_dir):
    """
    Saves trained PyTorch model inside 'models' dir. If dir doesn't exist dir created
    Parameters
    model : torch.nn.Module - Trained model to save
    name : str - Name for file
    Returns
    path : Path - Full path to saved file
    """
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f'{name}_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to: {model_path}')
    return model_path

def save_history(history, name, stage, model, model_dir, config=None):
    model_dir.mkdir(exist_ok=True)
    history_path = model_dir / f'{name}_{stage}_history.json'
    payload = {
        "model": name,
        "architecture": str(model),
        "stage": stage,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config or {},
        "metrics": history
    }
    with open(history_path, 'w') as f:
        json.dump(payload, f, indent=4)
    print(f'History saved to: {history_path}')
    return history_path

def full_train_old(name, images, labels, train_loader, val_loader, method, epochs, model_dir,
               config=None, dropout_prob=0.0, training_step=gru_step, save_outputs=True,
               session=None,**kwargs):
    from utils.training_session import create_training_session
    start_time = time.time()
    if session is None:
        session = create_training_session(images, labels, method, dropout_prob, config, training_step, **kwargs)
    session.train(epochs,train_loader,val_loader)
    model_path , history_path = None, None
    if save_outputs:
        model_path = save_model(session.model, name, model_dir)
        history_path = save_history(session.history, name, "train", session.model, model_dir, config=config)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n{name} training completed in {elapsed:.2f} seconds")
    return session.model, session.history, model_path, history_path

def full_train(name, builder, cfg, train_loader, val_loader, epochs, model_dir,
               training_step=gru_step, save_outputs=True, session=None):
    """
    General purpose training entry point.
    Args
    ----
    name          : Model name used for saved files.
    builder       : Function (cfg) -> (model, criterion, optimiser, training_kwargs).
                    Write one per model — see build_gru, build_lstm in network.py.
    cfg           : Full config dict passed to builder and saved with history.
    train_loader  : Training DataLoader.
    val_loader    : Validation DataLoader.
    epochs        : Number of epochs to train.
    model_dir     : Directory to save model and history.
    training_step : Training step function e.g. gru_step.
    save_outputs  : Whether to save model weights and history to disk.

    Returns
    -------
    (model, history, model_path, history_path)
    """
    from utils.training_session import TrainingSession
    start_time = time.time()

    if session is None:
        model, criterion, optimiser, training_kwargs = builder(cfg)
        session = TrainingSession(model=model,optimiser=optimiser,criterion=criterion,config=cfg,
                                  training_step=training_step,training_kwargs=training_kwargs)
    session.train(epochs, train_loader, val_loader)
    model_path, history_path = None, None
    if save_outputs:
        model_path   = save_model(session.model, name, model_dir)
        history_path = save_history(session.history, name, "train", session.model, model_dir, config=cfg)
    elapsed = time.time() - start_time
    print(f"\n{name} training completed in {elapsed:.2f} seconds")
    return session.model, session.history, model_path, history_path

def load_history(path: Path) -> dict:
    """
    Load a JSON history file saved by save_history() in utils/common.py.

    Args:
        path (Path): Path to the JSON history file.

    Returns:
        dict: Full JSON payload including metrics.
    """
    with open(path) as f:
        return json.load(f)


def extract_epoch_metrics(history: dict):
    """
    Pull per-epoch train/val accuracy from a loaded history dict.

    Args:
        history (dict): Loaded JSON history dict from load_history().

    Returns:
        tuple:
            epochs     (list[int])   — epoch numbers
            train_acc  (list[float]) — training accuracy per epoch
            val_acc    (list[float]) — validation accuracy per epoch
    """
    metrics   = history["metrics"]["epoch_metrics"]
    epochs    = [m["epoch"]               for m in metrics]
    train_acc = [m.get("train_accuracy")      for m in metrics]
    val_acc   = [m["validation_accuracy"] for m in metrics]
    return epochs, train_acc, val_acc


def load_model(dropout_prob: float, weights_path: Path) -> torch.nn.Module:
    """
    Instantiate a CNN and load saved weights from a .pt file.

    Args:
        dropout_prob  (float): Dropout probability used when the model was trained.
        weights_path  (Path):  Path to the saved state dict (.pt file).

    Returns:
        torch.nn.Module: Model with loaded weights in eval mode, on device.
    """
    model = SalesGRU(dropout=dropout_prob).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def evaluate_test_set(model, test_loader):
    """
    Evaluate a trained model on the test dataset.
    Args:
        model (torch.nn.Module)
        test_loader (DataLoader)
    Returns:
        dict containing test_loss and test_accuracy
    """
    criterion = nn.MSELoss()
    test_loss, test_acc, _, _ = evaluate_model(test_loader, model, criterion)
    print("\nTest performance")
    print(f"test_loss={test_loss:.4f}")
    print(f"test_acc={test_acc:.4f}")
    return {"test_loss": test_loss, "test_accuracy": test_acc}


def save_json(data, path):
    """
    Save dictionary as formatted JSON.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((preds - targets) ** 2)).item()

def mae(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.mean(torch.abs(preds - targets)).item()

def mape(preds, targets): #Mean Absolute Percentage Error
    mask = targets > 0  # avoid division by zero on zero sales days
    return ((preds[mask] - targets[mask]).abs() / targets[mask]).mean().item() * 100

def r2(preds, targets): #R Squared
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()

def nb_nll_loss(mu, alpha, targets):
    """
    Negative Binomial NLL loss.
    mu    : predicted mean   — must be > 0
    alpha : dispersion     — must be > 0
    targets: actual sales
    """
    eps = 1e-8
    r = 1.0 / (alpha + eps)
    p = mu / (mu + r + eps)
    nll = (
        -torch.lgamma(targets + r)
        + torch.lgamma(r)
        + torch.lgamma(targets + 1)
        - r * torch.log(r / (r + mu + eps))
        - targets * torch.log(p + eps)
    )
    return nll.mean()

def gaussian_nll_loss(mu, sigma, targets, sigma_reg=0.0):
    """
    Gaussian NLL loss in log1p space.

    mu      : predicted mean in log1p space
    sigma   : predicted std  in log1p space
    targets : log1p transformed sales
    sigma_reg: penalty on mean sigma to discourage variance collapse. — must be > 0
    """
    mu = torch.nan_to_num(mu, nan=0.0, posinf=10.0, neginf=-10.0)
    sigma = sigma.clamp(min=1e-6, max=10.0)
    dist  = torch.distributions.Normal(mu, sigma)
    nll = -dist.log_prob(targets).mean()
    if sigma_reg > 0.0:
        nll = nll + sigma_reg * sigma.mean()
    return nll
def pinball_loss(preds, targets, quantiles):
    """
    Pinball (quantile) loss.
    preds    : (batch, n_quantiles) — raw model output
    targets  : (batch,) or (batch, 1)
    quantiles: list of floats e.g. [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]

    Reduction: mean over batch, sum over quantiles — matches teammate's
    PinballLoss.forward() so loss scales are comparable across models.
    """
    if preds.dim() == 3:
        # seq2seq: preds (B, H, Q), targets (B, H)
        targets = targets.unsqueeze(2)  # (B, H, 1)
        q = torch.tensor(quantiles, dtype=torch.float32,
                         device=preds.device).view(1, 1, -1)  # (1, 1, Q)
        errors = targets - preds  # (B, H, Q)
        loss = torch.max((q - 1) * errors, q * errors)  # (B, H, Q)
        return loss.mean(dim=0).sum()  # mean over B, sum over H*Q
        # original autoregressive path — untouched
    if targets.dim() == 1:
        targets = targets.unsqueeze(1)  # (B, 1)
    q = torch.tensor(quantiles, dtype=torch.float32,
                     device=preds.device).unsqueeze(0)  # (1, Q)
    errors = targets - preds  # (B, Q)
    loss = torch.max((q - 1) * errors, q * errors)  # (B, Q)
    return loss.mean(dim=0).sum()


def weighted_pinball_loss(preds, targets, weights, quantiles):
    """
    Revenue-weighted pinball loss.
    preds    : (batch, n_quantiles)
    targets  : (batch,) or (batch, 1)
    weights  : (batch,) — per-item revenue weights, sum to 1 across dataset
    quantiles: list of floats

    Reduction: weighted sum over batch (weights sum to 1 so this is a
    weighted mean), then sum over quantiles. Matches teammate's fix —
    simple mean(dim=0) was wrong because it ignored the weight magnitudes;
    summing weighted losses gives the true revenue-weighted expectation.
    """
    if preds.dim() == 3:
        # seq2seq: preds (B, H, Q), targets (B, H), weights (B,)
        targets = targets.unsqueeze(2)  # (B, H, 1)
        weights = weights.view(-1, 1, 1)  # (B, 1, 1)
        q = torch.tensor(quantiles, dtype=torch.float32,
                         device=preds.device).view(1, 1, -1)  # (1, 1, Q)
        errors = targets - preds  # (B, H, Q)
        loss_per_q = torch.max((q - 1) * errors, q * errors)  # (B, H, Q)
        weighted_loss = loss_per_q * weights  # (B, H, Q)
        return weighted_loss.sum(dim=0).sum()  # weighted sum over B, sum over H*Q
        # original autoregressive path — untouched
    if targets.dim() == 1:
        targets = targets.unsqueeze(1)
    if weights.dim() == 1:
        weights = weights.unsqueeze(1)
    q = torch.tensor(quantiles, dtype=torch.float32,
                     device=preds.device).unsqueeze(0)
    errors = targets - preds
    loss_per_q = torch.max((q - 1) * errors, q * errors)
    weighted_loss = loss_per_q * weights
    return weighted_loss.sum(dim=0).sum()