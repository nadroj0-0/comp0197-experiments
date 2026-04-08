import torch


def nb_nll_loss(mu, alpha, targets):
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
    mu = torch.nan_to_num(mu, nan=0.0, posinf=10.0, neginf=-10.0)
    sigma = sigma.clamp(min=1e-6, max=10.0)
    dist = torch.distributions.Normal(mu, sigma)
    nll = -dist.log_prob(targets).mean()
    if sigma_reg > 0.0:
        nll = nll + sigma_reg * sigma.mean()
    return nll


def pinball_loss(preds, targets, quantiles):
    if preds.dim() == 3:
        targets = targets.unsqueeze(2)
        q = torch.tensor(quantiles, dtype=torch.float32, device=preds.device).view(1, 1, -1)
        errors = targets - preds
        loss = torch.max((q - 1) * errors, q * errors)
        return loss.mean(dim=0).sum()
    if targets.dim() == 1:
        targets = targets.unsqueeze(1)
    q = torch.tensor(quantiles, dtype=torch.float32, device=preds.device).unsqueeze(0)
    errors = targets - preds
    loss = torch.max((q - 1) * errors, q * errors)
    return loss.mean(dim=0).sum()


def weighted_pinball_loss(preds, targets, weights, quantiles):
    if preds.dim() == 3:
        targets = targets.unsqueeze(2)
        weights = weights.view(-1, 1, 1)
        q = torch.tensor(quantiles, dtype=torch.float32, device=preds.device).view(1, 1, -1)
        errors = targets - preds
        loss_per_q = torch.max((q - 1) * errors, q * errors)
        weighted_loss = loss_per_q * weights
        return weighted_loss.sum(dim=0).sum()
    if targets.dim() == 1:
        targets = targets.unsqueeze(1)
    if weights.dim() == 1:
        weights = weights.unsqueeze(1)
    q = torch.tensor(quantiles, dtype=torch.float32, device=preds.device).unsqueeze(0)
    errors = targets - preds
    loss_per_q = torch.max((q - 1) * errors, q * errors)
    weighted_loss = loss_per_q * weights
    return weighted_loss.sum(dim=0).sum()

