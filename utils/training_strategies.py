import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gru_step(model, inputs, labels, criterion, **kwargs):
    """
    Standard GRU training step for regression.
    Returns MSE loss and predictions.
    """
    outputs = model(inputs)
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
    loss = criterion(mu, sigma, labels, sigma_reg=kwargs.get("sigma_reg", 0.0))
    return loss, mu

prob_gru_step.valid_train_accuracy = False