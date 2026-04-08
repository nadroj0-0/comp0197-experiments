import time

import torch
import torch.nn as nn

from utils.training.early_stopping import EarlyStopping
from utils.training.strategies import gru_step

from .runtime import device
from .serialisation import save_history, save_model


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
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            all_preds.append(outputs.detach().cpu())
            all_targets.append(labels.detach().cpu())
    average_loss = total_loss / total_samples
    accuracy = None
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    return average_loss, accuracy, all_preds, all_targets


def train_model(
    epochs,
    train_loader,
    val_loader,
    model,
    criterion,
    optim_method,
    training_step=gru_step,
    early_stopping_patience=None,
    early_stopping_min_delta=0.0,
    scheduler=None,
    clip_grad_norm=None,
    extra_metrics=None,
    **kwargs,
):
    print("\nStarting training...")
    history: dict = {"epoch_metrics": [], "early_stopping": None, "batch_losses": []}
    accuracy_valid = getattr(training_step, "valid_train_accuracy", True)
    early_stopping_enabled = (
        early_stopping_patience is not None and early_stopping_patience > 0
    )
    early_stopper = (
        EarlyStopping(early_stopping_patience, min_delta=early_stopping_min_delta)
        if early_stopping_enabled
        else None
    )
    use_amp = device.type == "cuda" and not kwargs.get("disable_amp", False)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    for epoch in range(epochs):
        model.train()
        epoch_train_loss_sum = torch.tensor(0.0, device=device)
        epoch_train_correct = 0
        epoch_train_samples = 0

        for i, batch in enumerate(train_loader):
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
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                scaler.step(optim_method)
                scaler.update()
            else:
                loss, outputs = training_step(model, inputs, labels, criterion, **kwargs)
                loss.backward()
                if clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                optim_method.step()
            batch_size = labels.size(0)
            loss_value = loss.detach()
            history["batch_losses"].append(
                {"epoch": epoch + 1, "batch": i + 1, "loss": loss_value.item()}
            )
            epoch_train_loss_sum += loss_value * batch_size
            if accuracy_valid:
                predictions = outputs.argmax(dim=1)
                epoch_train_correct += (predictions == labels).sum().item()
            epoch_train_samples += batch_size

        train_loss = (epoch_train_loss_sum / epoch_train_samples).item()
        if accuracy_valid:
            train_accuracy = epoch_train_correct / epoch_train_samples
        eval_kwargs = {k: v for k, v in kwargs.items() if k != "item_weights"}
        val_loss, val_accuracy, val_preds, val_targets = evaluate_model(
            val_loader,
            model,
            criterion,
            training_step=training_step,
            **eval_kwargs,
        )
        if scheduler is not None:
            scheduler.step(val_loss)
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
        }
        if accuracy_valid:
            epoch_record["train_accuracy"] = train_accuracy
        if extra_metrics is not None:
            for name, fn in extra_metrics.items():
                epoch_record[name] = fn(val_preds, val_targets)
        history["epoch_metrics"].append(epoch_record)
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
    print("Training finished.")
    if early_stopper and early_stopper.stopped_epoch is None:
        early_stopper.stopped_epoch = epochs
    best_val_accuracy = None
    if early_stopper and early_stopper.best_model_state is not None:
        model.load_state_dict(early_stopper.best_model_state)
        print("Restored best model from early stopping.")
        print(
            f"Best validation loss {early_stopper.best_val_loss:.4f} "
            f"at epoch {early_stopper.best_epoch}"
        )
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
            "stopped_epoch": early_stopper.stopped_epoch,
        }
    return history
def full_train(
    name,
    builder,
    cfg,
    train_loader,
    val_loader,
    epochs,
    model_dir,
    training_step=gru_step,
    save_outputs=True,
    session=None,
):
    from utils.training.session import TrainingSession

    start_time = time.time()
    if session is None:
        model, criterion, optimiser, training_kwargs = builder(cfg)
        session = TrainingSession(
            model=model,
            optimiser=optimiser,
            criterion=criterion,
            config=cfg,
            training_step=training_step,
            training_kwargs=training_kwargs,
        )
    session.train(epochs, train_loader, val_loader)
    model_path, history_path = None, None
    if save_outputs:
        model_path = save_model(session.model, name, model_dir)
        history_path = save_history(
            session.history, name, "train", session.model, model_dir, config=cfg
        )
    elapsed = time.time() - start_time
    print(f"\n{name} training completed in {elapsed:.2f} seconds")
    return session.model, session.history, model_path, history_path
