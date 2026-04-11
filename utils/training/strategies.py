import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gru_step(model, inputs, labels, criterion, **kwargs):
    outputs = model(inputs)
    outputs = outputs.squeeze(-1)
    loss = criterion(outputs, labels)
    return loss, outputs


gru_step.valid_train_accuracy = False


def prob_gru_step(model, inputs, labels, criterion, **kwargs):
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
    quantiles = kwargs["quantiles"]
    preds = model(inputs)
    loss = criterion(preds, labels, quantiles)
    median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    if preds.dim() == 3:
        return loss, preds[:, :, median_idx]
    return loss, preds[:, median_idx]


quantile_gru_step.valid_train_accuracy = False


def wquantile_gru_step(model, inputs, labels, criterion, **kwargs):
    from utils.common import pinball_loss

    quantiles = kwargs["quantiles"]
    preds = model(inputs)
    if "item_weights" in kwargs:
        item_weights = kwargs["item_weights"]
        loss = criterion(preds, labels, item_weights, quantiles)
    else:
        loss = pinball_loss(preds, labels, quantiles)
    median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    if preds.dim() == 3:
        return loss, preds[:, :, median_idx]
    return loss, preds[:, median_idx]


wquantile_gru_step.valid_train_accuracy = False

