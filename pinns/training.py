from __future__ import annotations

import numpy as np
import torch
from typing import Optional, Tuple, Callable, Type, TYPE_CHECKING

if TYPE_CHECKING:
    import optuna
    from .parser import AttrDict
    from .density_estimation import DensityEstimator
    from .data_loader import DataPlaceholder


def check_model_hp(hp: AttrDict) -> AttrDict:
    """Set default values for model-related hyperparameters.

    Defaults: linear='HE', eps=1e-8, clip_gradients=True,
    cosine_annealing=off, relobralo=off.

    Parameters
    ----------
    hp : AttrDict

    Returns
    -------
    AttrDict
        hp with defaults filled in.
    """
    if "linear" not in hp.model:
        hp.model["linear"] = "HE"
    if "eps" not in hp:
        hp.eps = 1.0e-8
    if "clip_gradients" not in hp:
        hp.clip_gradients = True
    if "cosine_annealing" not in hp:
        hp.cosine_annealing = {"status": False, "min_eta": 0, "step": 500}
    if "relobralo" not in hp:
        hp.relobralo = {
            "status": False,
            "T": 1.0,
            "alpha": 0.999,
            "rho": 0.5,
            "step": 100,
        }
    return hp


def check_data_loader_hp(hp: AttrDict) -> AttrDict:
    """Set default values for data-loader-related hyperparameters.

    Defaults: model.name='default', hard_periodicity=False.

    Parameters
    ----------
    hp : AttrDict

    Returns
    -------
    AttrDict
    """
    if "model" in hp:
        if "name" not in hp.model:
            hp.model["name"] = "default"
    else:
        hp.model = {"name": "default"}
    if "hard_periodicity" not in hp:
        hp.hard_periodicity = False
    return hp


def check_estimator_hp(hp: AttrDict) -> AttrDict:
    """Set default values for estimator-related hyperparameters.

    Defaults: verbose=True, optuna patience/trials, save_model=False.

    Parameters
    ----------
    hp : AttrDict

    Returns
    -------
    AttrDict
    """
    if "verbose" not in hp:
        hp.verbose = True
    if "optuna" not in hp:
        hp.optuna = {"patience": 10000, "trials": 1}
    if "save_model" not in hp:
        hp.save_model = False
    # if "npz_name" not in hp:
    #     hp.npz_name = "default.npz"
    # if "pth_name" not in hp:
    #     hp.pth_name = "default.pth"
    return hp


def train(
    hp: AttrDict,
    estimate_density_cl: Type[DensityEstimator],
    dataset_fn: Callable[[AttrDict, bool], Tuple[DataPlaceholder, DataPlaceholder]],
    model_cl: Callable,
    initial_weights: Optional[str] = None,
    trial: Optional[optuna.Trial] = None,
    gpu: bool = False,
) -> Tuple[DensityEstimator, AttrDict]:
    """High-level training function for PINN models.

    Orchestrates data loading, model creation, and training. Mutates
    hp to add input_size, output_size, nv_samples, and nv_targets.

    Parameters
    ----------
    hp : AttrDict
        Hyperparameter dictionary loaded from YAML.
    estimate_density_cl : class
        A DensityEstimator subclass implementing domain-specific losses.
    dataset_fn : callable
        Function(hp, gpu=False) -> (train_dataset, test_dataset).
    model_cl : class or callable
        Model constructor: model_cl(name, input_size, output_size, hp) -> nn.Module.
    initial_weights : str, optional
        Path to .pth file for weight initialisation. Default: None.
    trial : optuna.Trial, optional
        Optuna trial for hyperparameter search. Default: None.
    gpu : bool, optional
        Whether to use GPU. Default: False.

    Returns
    -------
    NN : DensityEstimator
        Trained estimator with model, loss history, etc.
    hp : AttrDict
        Updated hyperparameters (with input/output sizes, normalisation values).
    """
    hp = check_data_loader_hp(hp)
    hp = check_model_hp(hp)
    hp = check_estimator_hp(hp)

    train, test = dataset_fn(hp, gpu=gpu)

    hp.input_size = train.input_size
    hp.output_size = train.output_size
    hp.nv_samples = train.nv_samples
    hp.nv_targets = train.nv_targets

    model = model_cl(
        hp.model["name"],
        hp.input_size,
        output_size=hp.output_size,
        hp=hp,
    )
    # model = torch.compile(model, backend="inductor")
    if gpu:
        model = model.cuda()

    NN = estimate_density_cl(train, test, model, hp, gpu, trial=trial)
    if initial_weights:
        NN.model.load_state_dict(torch.load(initial_weights, map_location=NN.device))
    NN.fit()

    if hp.save_model:
        np.savez(
            hp.npz_name,
            **hp,
        )

    return NN, hp
