"""
Training package.

Import concrete submodules directly, e.g.:
- utils.training.strategies
- utils.training.session
- utils.training.hyperparameter

Keeping this package init lightweight avoids circular imports with utils.common.
"""

from .early_stopping import EarlyStopping
from .optimisation import OptimisationConfig
