"""INR4torch: Physics-Informed Neural Networks with Implicit Neural Representations."""

__version__ = "0.1.0"

__all__ = [
    "data_loader",
    "density_estimation",
    "models",
    "training",
    "parser",
    "diff_operators",
    "pde_utils",
]

from .data_loader import DataPlaceholder
from .density_estimation import DensityEstimator
from .models import INR
from .parser import AttrDict, read_yaml
from .training import train
from .diff_operators import hessian, laplace, divergence, gradient, jacobian
from .pde_utils import gen_uniform
