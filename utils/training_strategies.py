import torch.nn.functional as F
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gru_step(model, inputs, labels, criterion, **kwargs):
    """
    Standard GRU training step for regression.
    Returns MSE loss and predictions.
    """
    outputs = model(inputs)
    outputs = outputs.squeeze(-1)
    loss = criterion(outputs, labels)
    return loss, outputs

gru_step.valid_train_accuracy = False

def prob_gru_step(model, inputs, labels, criterion, **kwargs):
    """
    Training step for probabilistic GRU/LSTM.
    """
    # mu, alpha = model(inputs)
    # loss = criterion(mu, alpha, labels)
    # return loss, mu  # return mu as predictions for metric computation
    mu, sigma = model(inputs)
    mu = mu.squeeze(-1)
    sigma = sigma.squeeze(-1)
    loss = criterion(mu, sigma, labels, sigma_reg=kwargs.get("sigma_reg", 0.0))
    return loss, mu

prob_gru_step.valid_train_accuracy = False

def prob_nb_step(model, inputs, labels, criterion, **kwargs):
    mu, alpha = model(inputs)
    mu = mu.squeeze(-1)
    alpha = alpha.squeeze(-1)
    loss = criterion(mu, alpha, labels)
    return loss, mu

prob_nb_step.valid_train_accuracy = False


def quantile_gru_step(model, inputs, labels, criterion, **kwargs):
    """
    Training step for quantile GRU (unweighted pinball loss).
    Returns median quantile as outputs so extra_metrics (rmse, mae etc.)
    receive a scalar point forecast per sample, not all 7 quantiles.
    """
    quantiles = kwargs["quantiles"]
    preds = model(inputs)  # (B, Q) or (B, H, Q)
    loss = criterion(preds, labels, quantiles)
    median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    if preds.dim() == 3:
        return loss, preds[:, :, median_idx]  # (B, H) — seq2seq median
    return loss, preds[:, median_idx]  # (B,)   — autoregressive unchanged

quantile_gru_step.valid_train_accuracy = False


def wquantile_gru_step(model, inputs, labels, criterion, **kwargs):
    """
    Training step for revenue-weighted quantile GRU (weighted pinball loss).
    Falls back to unweighted pinball during val/test when item_weights absent.
    Returns median quantile as outputs so extra_metrics receive a scalar
    point forecast per sample.
    """
    from utils.common import pinball_loss
    quantiles  = kwargs["quantiles"]
    preds      = model(inputs)                                  # (B, Q)
    if "item_weights" in kwargs:
        item_weights = kwargs["item_weights"]
        loss = criterion(preds, labels, item_weights, quantiles)
    else:
        loss = pinball_loss(preds, labels, quantiles)
    median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    if preds.dim() == 3:
        return loss, preds[:, :, median_idx]  # (B, H)
    return loss, preds[:, median_idx]  # (B,)

wquantile_gru_step.valid_train_accuracy = False


def tft_step(model, inputs, labels, criterion, **kwargs):
    """
    Training step for the Temporal Fusion Transformer.

    inputs : dict of tensors — the full TFT batch (encoder_cont, decoder_cat, etc.)
             already moved to device by move_to_device() in common.py train_model.
    labels : (target_tensor, weight) tuple from TimeSeriesDataSet loader.
             target_tensor() extracts the (B, H) target; weight is None for TFT.
    criterion : QuantileLoss from pytorch-forecasting.
    kwargs must contain "quantiles" (list of floats) — set by build_tft().

    Returns (loss scalar, median_predictions of shape (B, H)) so that
    extra_metrics (rmse, mae etc.) in common.py receive a point forecast.
    """
    from utils.common import target_tensor
    out = model(inputs)  # inputs is the dict, on device
    pred = out["prediction"]  # (B, H, Q)
    target = target_tensor(labels)  # (B, H) — unwrap (tensor, None)
    loss = criterion.loss(pred, target).mean()
    quantiles = kwargs["quantiles"]  # required — no default
    median_idx = quantiles.index(0.5)
    return loss, pred[:, :, median_idx]  # (B, H) median as point pred


tft_step.valid_train_accuracy = False
