from utils.common import init_loss, init_model, init_optimiser, train_model


class TrainingSession:
    def __init__(
        self,
        model,
        optimiser,
        criterion,
        config,
        training_step,
        training_kwargs=None,
        history=None,
        epoch=0,
    ):
        self.model = model
        self.optimiser = optimiser
        self.criterion = criterion
        self.config = config
        self.training_step = training_step
        self.training_kwargs = training_kwargs or {}
        self.history = history or {
            "epoch_metrics": [],
            "batch_losses": [],
            "early_stopping": None,
        }
        self.epoch = epoch

    def train(self, epochs, train_loader, val_loader):
        history = train_model(
            epochs=epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            model=self.model,
            criterion=self.criterion,
            optim_method=self.optimiser,
            training_step=self.training_step,
            early_stopping_patience=self.config.get("early_stopping_patience") if self.config else None,
            early_stopping_min_delta=self.config.get("early_stopping_min_delta") if self.config else 0.0,
            **self.training_kwargs,
        )
        for metric in history["epoch_metrics"]:
            metric["epoch"] += self.epoch
        for batch in history["batch_losses"]:
            batch["epoch"] += self.epoch
        self.history["epoch_metrics"].extend(history["epoch_metrics"])
        self.history["batch_losses"].extend(history["batch_losses"])
        if history.get("early_stopping") is not None:
            self.history["early_stopping"] = history["early_stopping"]
        self.epoch += len(history["epoch_metrics"])
        return history


def create_training_session(images, labels, method, dropout_prob, config, training_step, **kwargs):
    model, outputs = init_model(images, dropout_prob)
    criterion, _ = init_loss(outputs, labels)
    optimiser_keys = {"lr", "momentum", "weight_decay", "dampening", "nesterov"}
    optimiser_kwargs = {k: v for k, v in kwargs.items() if k in optimiser_keys}
    non_training_keys = {
        "epochs",
        "batch_size",
        "seed",
        "optimiser",
        "validation_fraction",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "reg_dropout",
    }
    training_kwargs = {
        k: v for k, v in kwargs.items()
        if k not in optimiser_keys and k not in non_training_keys
    }
    optimiser = init_optimiser(model, method, **optimiser_kwargs)
    if config:
        for key in ["mixup_alpha", "label_smoothing"]:
            if key in config:
                training_kwargs[key] = config[key]
    return TrainingSession(model, optimiser, criterion, config, training_step, training_kwargs)

