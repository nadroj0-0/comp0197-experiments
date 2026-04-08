from torch.utils.data import DataLoader
import torch
from pathlib import Path
from .common import *
from .training.hyperparameter import staged_search


def get_model_dir(exp_name, base_dir=None):
    if base_dir is None:
        base_dir = Path(__file__).parent
    model_dir = base_dir / "models" / exp_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir

def run_test_evaluation(model, test_dataset, batch_size, name, model_dir,config=None):
    """
    Complete test evaluation pipeline.
    Builds test loader → evaluates model → attaches metrics → saves history.
    Args:
        model (torch.nn.Module)
        test_dataset
        batch_size (int)
        history (dict)
        experiment_name (str)
        model_dir (Path)
        config (dict)
    Returns:
        dict: test metrics
    """
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_metrics = evaluate_test_set(model, test_loader)
    history_path = save_history(test_metrics, name, "test", model, model_dir, config=config)
    return test_metrics, history_path

def evaluate_confidence(model, data_loader):
    """
    Compute mean max softmax confidence across a dataset.
    A well-calibrated model produces lower confidence than an overfit one.

    Args:
        model       (torch.nn.Module): Trained model in eval mode.
        data_loader (DataLoader):      Dataset to evaluate over.

    Returns:
        float: Mean of the maximum softmax probability across all samples.
    """
    model.eval()
    total_confidence = 0.0
    total_samples    = 0
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs   = inputs.to(device)
            logits   = model(inputs)
            probs    = torch.softmax(logits, dim=1)
            max_prob = probs.max(dim=1).values
            total_confidence += max_prob.sum().item()
            total_samples    += inputs.size(0)
    return total_confidence / total_samples

class Experiment:
    """
    Encapsulates a single training experiment: data loading, optional
    hyperparameter search, training, and evaluation.
    """
    def __init__(self, name, base_cfg, model_dir=None):
        self.name = name
        self.cfg = base_cfg.copy()
        self.model_dir = model_dir or get_model_dir(name)
        self.model = None
        self.history = None
        self.stats = None
        self.preloaded = False


    def prepare_data(self, data_fn, **data_kwargs):
        """
        Load data using data_fn and store as instance attributes.
        data_fn must return (train_loader, val_loader, test_dataset, stats).
        stats can be None for problems without normalisation e.g. CW1.
        """
        self.train_loader, self.val_loader, self.test_dataset, self.stats = data_fn(**data_kwargs)


    def search(self, search_space, builder, training_step=gru_step,
               schedule=None, initial_models=20):
        """Run successive halving search and update self.cfg with winner."""
        from utils.training.hyperparameter import staged_search
        print(f"\nStarting {self.name} hyperparameter search")
        best_cfg = staged_search(
            search_space,
            self.train_loader, self.val_loader,
            builder,
            self.model_dir,
            base_config=self.cfg,
            training_step=training_step,
            schedule=schedule,
            initial_models=initial_models,
            search_name=f"{self.name}_search",
        )
        if best_cfg is not None:
            print(f"Best {self.name} configuration:")
            print(best_cfg)
            self.cfg = best_cfg.copy()


    def train(self, builder, training_step=gru_step):
        """Train the final model using self.cfg."""
        init_seed(self.cfg)
        self.model, self.history, _, _ = full_train(
            self.name, builder, self.cfg,
            self.train_loader, self.val_loader,
            self.cfg["epochs"], self.model_dir,
            training_step=training_step,
        )
        print(f'\n{self.name} model:')
        print(self.model)
        print(f'\n{self.name} final epoch metrics:')
        print(self.history['epoch_metrics'][-1])


    def run(self, builder, training_step=gru_step,
            data_fn=None, search_space=None, schedule=None, initial_models=20,
            **data_kwargs):
        """
        Run the full experiment pipeline in one call:
        data preparation -> optional search -> training.
        Args
        ----
        data_fn       : Function returning (train_loader, val_loader, test_dataset, stats).
        builder       : Function (cfg) -> (model, criterion, optimiser, training_kwargs).
        training_step : Training step function e.g. gru_step, baseline_step.
        search_space  : Dict of hyperparameter ranges. None skips search.
        schedule      : Successive halving schedule. None uses default.
        initial_models: Number of configs to sample initially.
        **data_kwargs : Passed directly to data_fn.
        """
        if not self.preloaded:
            self.prepare_data(data_fn, **data_kwargs)
        if search_space is not None:
            self.search(search_space, builder, training_step, schedule, initial_models)
        self.train(builder, training_step)
