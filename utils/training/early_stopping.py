import copy


class EarlyStopping:
    """
    Validation-based early stopping utility.
    Tracks the best validation loss and stops training when
    it has not improved for 'patience' epochs.
    """

    def __init__(self, patience: int, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float("inf")
        self.best_epoch = None
        self.best_model_state = None
        self.counter = 0
        self.stopped_epoch = None
        self.triggered = False

    def update(self, val_loss, model, epoch):
        """
        Update early stopping state for a new epoch.
        Returns True if training should stop.
        """
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.stopped_epoch = epoch
            return True
        return False

