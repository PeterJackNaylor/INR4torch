import numpy as np
from tqdm import trange, tqdm
import optuna
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


def Norm1(vector, dim=None):
    """Compute the L1 norm of a vector.

    Parameters
    ----------
    vector : torch.Tensor
        Input tensor.
    dim : int, optional
        Dimension along which to compute.

    Returns
    -------
    torch.Tensor
    """
    return torch.linalg.vector_norm(vector, ord=1, dim=dim)


def Norm2(vector, dim=None):
    """Compute the L2 norm of a vector.

    Parameters
    ----------
    vector : torch.Tensor
        Input tensor.
    dim : int, optional
        Dimension along which to compute.

    Returns
    -------
    torch.Tensor
    """
    return torch.linalg.vector_norm(vector, ord=2, dim=dim)


class WL1Loss(nn.Module):
    """Weighted L1 (MAE) loss.

    Parameters
    ----------
    ignore_nan : bool, optional
        Currently unused. Default: False.
    """

    def __init__(self, ignore_nan=False):
        super().__init__()
        self.mae = nn.L1Loss()

    def forward(self, zhat, z, **args_dic):

        if "weight" not in args_dic:
            loss = self.mae(zhat, z)
        else:
            weight = args_dic["weight"]
            loss = torch.absolute(weight * (zhat - z)).mean()
        return loss


class RMSELoss(nn.Module):
    """Root Mean Squared Error loss with optional weighting and NaN handling.

    Parameters
    ----------
    eps : float, optional
        Small constant for numerical stability. Default: 1e-6.
    column : int, optional
        If set, compute loss only on this output column. Default: None.
    ignore_nan : bool, optional
        If True, ignore NaN values in targets. Default: False.
    """

    def __init__(self, eps=1e-6, column=None, ignore_nan=False):
        super().__init__()
        mean_f = torch.mean if not ignore_nan else torch.nanmean
        if not ignore_nan:

            def mse(yhat, y):
                return mean_f((yhat - y) ** 2)

            def wmse(yhat, y, w):
                return mean_f(w * (yhat - y) ** 2)

        else:

            def mse(yhat, y):
                nan = torch.isnan(y)
                y = torch.where(nan, torch.tensor(0.0), y)
                return mean_f((yhat - y) ** 2)

            def wmse(yhat, y, w):
                nan = torch.isnan(y)
                y = torch.where(nan, torch.tensor(0.0), y)
                return mean_f(w * (yhat - y) ** 2)

        self.mse = mse
        self.wmse = wmse
        self.eps = eps
        self.column = column

    def forward(self, zhat, z, **args_dic):
        if self.column is not None:
            zhat = zhat[:, self.column]
            z = z[:, self.column]
        if "weight" not in args_dic:
            loss = self.mse(zhat, z)
        else:
            weight = args_dic["weight"]
            if weight is None:
                loss = self.mse(zhat, z)
            else:
                loss = self.wmse(zhat, z, weight)

        return torch.sqrt(loss + self.eps)


def checkpoint_values(dic_values, dic_tmp_values):
    """Aggregate temporary per-step loss values into the main checkpoint dict.

    Parameters
    ----------
    dic_values : dict of list
        Main loss history (each key maps to a list of scalars).
    dic_tmp_values : dict of list
        Temporary per-step tensors to be averaged and appended.
    """
    for key in dic_values.keys():
        dic_values[key].append(torch.stack(dic_tmp_values[key]).mean().item())


def model_params_data(model, hp):
    """Yield gradient data for all model parameters that have gradients.

    Parameters
    ----------
    model : nn.Module
    hp : AttrDict

    Yields
    ------
    torch.Tensor
        Gradient data (.grad.data) for each parameter.
    """
    params = model.parameters()
    for p in params:
        try:
            data = p.grad.data
            yield data
        except AttributeError:
            pass


def balancing_loss(loss_values, lambdas, alpha, model, hp, device):
    """Self-adapting loss balancing based on gradient norms.

    Updates lambda weights for each loss term proportionally to the
    inverse of their gradient norms, with exponential moving average.

    Parameters
    ----------
    loss_values : dict
        Loss name -> list of loss tensors.
    lambdas : dict
        Loss name -> list of lambda weights.
    alpha : float
        Exponential moving average coefficient (0 < alpha < 1).
    model : nn.Module
        The neural network (for gradient computation).
    hp : AttrDict
        Hyperparameters.
    device : str
        'cuda' or 'cpu'.
    """
    grad_norms = {}
    keys = loss_values.keys()
    keys = [k for k in keys if hp.losses[k]["loss_balancing"]]
    for k in keys:
        if lambdas[k][-1] != 0:
            loss_k = loss_values[k][-1]
            loss_k.backward(retain_graph=True)
            grad_loss_k = torch.cat(
                [p.flatten().clone() for p in model_params_data(model, hp)]
            )
            grad_norms[k] = Norm2(grad_loss_k)
            if device != "cpu":
                grad_norms[k] = grad_norms[k].cpu()
            grad_norms[k] = grad_norms[k]
            for p in model_params_data(model, hp):
                p.zero_()
        else:
            grad_norms[k] = 0
    summed_grads = np.sum(list(grad_norms.values()))
    for k in keys:
        if lambdas[k][-1] != 0:
            if grad_norms[k] == 0:
                new_val = lambdas[k][-1]
            else:
                new_val = (summed_grads / grad_norms[k]).item()
                new_val = lambdas[k][-1] * alpha + new_val * (1 - alpha)
        else:
            new_val = 0
        lambdas[k].append(new_val)


class DensityEstimator:
    """Base training class for physics-informed neural networks.

    Manages the full training loop including optimiser setup, learning
    rate scheduling, loss balancing (self-adapting / ReLoBRaLo),
    temporal causality weighting, early stopping, and Optuna pruning.

    Subclasses should implement domain-specific PDE loss methods.

    Parameters
    ----------
    train : DataPlaceholder
        Training dataset (also serves as iterator).
    test : DataPlaceholder
        Test dataset.
    model : nn.Module
        The neural network model.
    model_hp : AttrDict
        Hyperparameters dictionary.
    gpu : bool
        Whether to use GPU.
    trial : optuna.Trial, optional
        Optuna trial for hyperparameter search. Default: None.

    Attributes
    ----------
    loss_values : dict
        Dictionary of loss name -> list of loss values per iteration.
    lambdas_scalar : dict
        Dictionary of loss name -> list of loss weights over time.
    test_scores : list
        Validation scores recorded every ``test_frequency`` iterations.
    """

    def __init__(self, train, test, model, model_hp, gpu, trial=None):
        self.data = train
        self.test_set = test
        self.n = len(train)
        self.n_test = len(test)
        self.n_outputs = self.data.output_size

        self.model = model
        self.hp = model_hp
        self.device = "cuda" if gpu else "cpu"
        self.trial = trial
        self.autocasting()

    def setup_optimizer(self):
        """Instantiate the optimizer (Adam, AdamW, or LBFGS) from hp config."""
        if self.hp.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.hp.lr,
                eps=self.hp.eps,
            )
        elif self.hp.optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.hp.lr,
                eps=self.hp.eps,
            )
        elif self.hp.optimizer == "LBFGS":
            self.optimizer = torch.optim.LBFGS(
                self.model.parameters(), lr=self.hp.lr, tolerance_change=self.hp.eps
            )
        else:
            raise NameError("Optimiser not implemented")

    def setup_scheduler(self):
        """Instantiate the learning rate scheduler.

        Supports StepLR decay and cosine annealing with warm restarts.
        """
        scheduler_status = True
        if self.hp.learning_rate_decay["status"]:
            step_size = self.hp.learning_rate_decay["step"]
            scheduler = StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=self.hp.learning_rate_decay["gamma"],
            )
            scheduler_status = True
        elif self.hp.cosine_annealing["status"]:
            self.steps = self.hp.cosine_annealing["step"]
            self.min_eta = self.hp.cosine_annealing["min_eta"]
            self.times = 0
            scheduler = CosineAnnealingLR(
                self.optimizer, self.steps, eta_min=self.min_eta
            )

        else:
            scheduler_status = False
        self.scheduler_status = scheduler_status
        if scheduler_status:
            self.lr_list = []
            self.scheduler = scheduler

    def setup_losses(self):
        """Build loss functions and lambda weights from hp.losses config.

        Handles multiple-output expansion, temporal causality,
        and custom PDE loss methods.
        """
        lambdas = {}
        loss_values = {}
        loss_fn = {}
        if self.n_outputs > 1:
            key_list = list(self.hp.losses.keys())
            for key in key_list:
                if "multiple_outputs" in self.hp.losses[key].keys():
                    if self.hp.losses[key]["multiple_outputs"]:
                        for i in range(self.n_outputs):
                            self.hp.losses[f"{key}_{i}"] = self.hp.losses[key].copy()
                            self.hp.losses[f"{key}_{i}"]["method"] = f"{key}_{i}"

                            setattr(
                                self,
                                f"{key}_{i}",
                                RMSELoss(
                                    column=i,
                                    ignore_nan=self.hp.losses[key]["ignore_nan"],
                                ),
                            )
                    del self.hp.losses[key]
                else:
                    self.hp.losses[key]["multiple_outputs"] = False

        for key in self.hp.losses.keys():
            if "lambda" in self.hp.losses[key]:
                lambdas[key] = [self.hp.losses[key]["lambda"]]
            else:
                lambdas[key] = [1]
            if key == "mse":
                if "ignore_nan" not in self.hp.losses[key].keys():
                    self.hp.losses[key]["ignore_nan"] = False
                loss_fn[key] = RMSELoss(ignore_nan=self.hp.losses[key]["ignore_nan"])
            elif key == "mae":
                loss_fn[key] = WL1Loss()
            else:
                if "method" in self.hp.losses[key]:
                    loss_computation_fn = getattr(self, self.hp.losses[key]["method"])
                    try:
                        penalty_fn = getattr(self, self.hp.losses[key]["penalty"])
                    except (AttributeError, KeyError):
                        penalty_fn = self.L2
                    bs_loss = self.hp.losses[key]["bs"]
                    if "temporal_causality" not in self.hp.losses[key].keys():
                        self.hp.losses[key]["temporal_causality"] = False
                    if self.hp.losses[key]["temporal_causality"]:
                        loss_fn[key] = self.temporal_loss(
                            key, loss_computation_fn, bs_loss, penalty_fn
                        )
                    else:
                        loss_fn[key] = self.standard_loss(
                            key, loss_computation_fn, bs_loss, penalty_fn
                        )
            loss_values[key] = []
        self.lambdas_scalar = lambdas
        self.loss_values = loss_values
        # for key in self.hp.losses.keys():
        #     loss_fn[key] = torch.compile(loss_fn[key])
        self.loss_fn = loss_fn
        self.test_scores = []

    def sum_loss(self, dic_values, dic_lambdas):
        """Compute weighted sum of all loss terms.

        Parameters
        ----------
        dic_values : dict
            Loss name -> list of loss tensors.
        dic_lambdas : dict
            Loss name -> list of lambda weights.

        Returns
        -------
        torch.Tensor
            Scalar total loss.
        """
        loss = 0.0
        for key in dic_lambdas.keys():
            loss += dic_lambdas[key][-1] * dic_values[key][-1]

        if len(dic_values[list(dic_values.keys())[0]]) > 1:
            for key in dic_lambdas.keys():
                dic_values[key][-2] = dic_values[key][-2].detach().item()
        return loss

    def setup_temporal_causality(self):
        """Initialise temporal causality weights for PDE losses.

        Reads M and eps from hp.temporal_causality config.
        """
        self.temporal_weights = {}
        for key in self.hp["losses"].keys():
            if "temporal_causality" in self.hp["losses"][key]:
                if self.hp["losses"][key]["temporal_causality"]:
                    M = self.hp.temporal_causality["M"]
                    self.temporal_weights[key] = []
                    self.M = M
                    self.eps = self.hp.temporal_causality["eps"]

    # def temporal_loss_data_fit(self, key, residue_computation, bs_loss, penalty):
    #     def f(zhat, z, weight=None, groups=None, **args_dic):
    #         residue = residue_computation(z, zhat, weight)
    #         M_losses = torch.zeros(self.M)
    #         for i in range(self.M):
    #             M_losses[i] = penalty(residue[groups == i], normed=True)
    #         f = self.hp.temporal_causality["step"]
    #         if self.it == 1 or (self.it - 1) % f == 0:
    #             shifted_M_loss = torch.zeros_like(M_losses)
    #             shifted_M_loss[1:] = M_losses[:-1]
    #             w_i = torch.exp(
    #                 -self.eps * torch.cumsum(shifted_M_loss, dim=0)
    #             ).detach()
    #             w_i.requires_grad_(False)
    #             self.temporal_weights[key].append(w_i)
    #         else:
    #             w_i = self.temporal_weights[key][-1]
    #         loss = torch.mean(w_i * M_losses)
    #         return loss

    #     return f

    def temporal_loss(self, key, residue_computation, bs_loss, penalty):
        """Build a temporal-causality-weighted loss closure.

        Splits PDE residuals into M temporal bins and applies
        exponentially decaying causal weights.

        Parameters
        ----------
        key : str
            Loss name (used to store temporal weights).
        residue_computation : callable
            Function computing the PDE residual.
        bs_loss : int
            Batch size for this loss (must be divisible by M).
        penalty : callable
            Norm function (L1 or L2).

        Returns
        -------
        callable
            Loss function with signature f(zhat, z, **kwargs) -> scalar.
        """

        def f(zhat, z, weight=None, **args_dic):
            residue = residue_computation(z, zhat, weight)
            square_res = residue.reshape(self.M, bs_loss // self.M)
            M_losses = penalty(square_res, dim=1, normed=False)
            f = self.hp.temporal_causality["step"]
            with torch.no_grad():
                if self.it == 1 or (self.it - 1) % f == 0:
                    shifted_M_loss = torch.zeros_like(M_losses)
                    shifted_M_loss[1:] = M_losses[:-1]
                    w_i = torch.exp(-self.eps * torch.cumsum(shifted_M_loss, dim=0))
                    w_i.requires_grad_(False)
                    self.temporal_weights[key].append(w_i)
                else:
                    w_i = self.temporal_weights[key][-1]
            loss = torch.mean(w_i * M_losses)
            return loss

        return f

    def compute_loss(self, zhat, **args_dic):
        """Evaluate all loss functions and append results to loss_values.

        Parameters
        ----------
        zhat : torch.Tensor
            Model predictions.
        **args_dic : dict
            Additional arguments (z=targets, x=inputs, weight=..., etc.).
        """
        for key in self.hp.losses.keys():
            self.loss_values[key].append(self.loss_fn[key](zhat, **args_dic))

    def L1(self, vector, normed=True, dim=None):
        """Compute (optionally normalised) L1 norm.

        Parameters
        ----------
        vector : torch.Tensor
        normed : bool
            If True, divide by the number of elements.
        dim : int, optional
        """
        result = Norm1(vector, dim=dim)
        if normed:
            if dim is None:
                dim = 0
            result = result / vector.shape[dim]
        return result

    def L2(self, vector, normed=True, dim=None):
        """Compute (optionally normalised) L2 norm.

        Parameters
        ----------
        vector : torch.Tensor
        normed : bool
            If True, divide by sqrt(n_elements).
        dim : int, optional
        """
        result = Norm2(vector, dim=dim)
        if normed:
            if dim is None:
                dim = 0
            result = result / (vector.shape[dim]) ** 0.5
        return result

    def standard_loss(self, key, residue_computation, bs_loss, penalty):
        """Build a standard (non-temporal) PDE loss closure.

        Parameters
        ----------
        key : str
            Loss name.
        residue_computation : callable
            Function computing the PDE residual.
        bs_loss : int
            Batch size for this loss.
        penalty : callable
            Norm function (L1 or L2).

        Returns
        -------
        callable
            Loss function with signature f(zhat, z, **kwargs) -> scalar.
        """

        def f(zhat, z, weight=None, **args_dict):
            residue = residue_computation(zhat, z, weight=weight).flatten()
            loss = penalty(residue, normed=True)
            return loss

        return f

    def range(self, start, end, step, leave=True):
        iterator_fn = trange if self.hp.verbose else range
        if not leave and self.hp.verbose:
            return iterator_fn(start, end, step, leave=leave)
        return iterator_fn(start, end, step)

    def loss_balancing(self):
        """Update loss lambda weights using the configured balancing strategy.

        Supports self-adapting loss balancing (gradient-norm based)
        and ReLoBRaLo (relative loss balancing with random lookback).
        """
        if len(self.lambdas_scalar.keys()) == 1:
            pass
        elif self.hp.self_adapting_loss_balancing["status"]:
            f = self.hp.self_adapting_loss_balancing["step"]
            if self.it == 1 or self.it % f == 0:
                balancing_loss(
                    self.loss_values,
                    self.lambdas_scalar,
                    self.hp.self_adapting_loss_balancing["alpha"],
                    self.model,
                    self.hp,
                    self.device,
                )
        elif self.hp.relobralo["status"]:
            f = self.hp.relobralo["step"]
            if self.it == 1 or self.it % f == 0:
                rho = self.hp.relobralo["rho"]
                alpha = self.hp.relobralo["alpha"]
                T = self.hp.relobralo["T"]
                with torch.no_grad():
                    sm = nn.Softmax(dim=0)

                    def g(li, i):
                        return li[i][-1] / (li[i][-2] * T + 1e-12)

                    def g0(li, i):
                        return li[i][-1] / (li[i][0] * T + 1e-12)

                    keys = self.loss_values.keys()
                    keys = [k for k in keys if self.hp.losses[k]["loss_balancing"]]

                    val = torch.stack([g(self.loss_values, i) for i in keys])
                    lambs_hat = sm(val) * len(keys)
                    val_0 = torch.stack([g0(self.loss_values, i) for i in keys])
                    lambs0_hat = sm(val_0) * len(keys)
                    for i, key in enumerate(keys):
                        l_i = (
                            rho * alpha * self.lambdas_scalar[key][-1]
                            + (1 - rho) * alpha * lambs0_hat[i]
                            + (1 - alpha) * lambs_hat[i]
                        )
                        if self.device == "cuda":
                            l_i = l_i.cpu()
                        self.lambdas_scalar[key].append(l_i.item())

    def clip_gradients(self):
        """Clip gradient norms to max_norm=1.0 if enabled in config."""
        if self.hp.clip_gradients:
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

    def scheduler_update(self):
        """Step the learning rate scheduler.

        For cosine annealing with warm restarts, halves the learning
        rate and minimum eta after each full cycle.
        """
        self.lr_list.append(self.scheduler.get_last_lr()[0])
        self.scheduler.step()

        if self.hp.cosine_annealing["status"]:
            # it starts on the end of the it number
            first_it = self.it == 1
            it_after = (self.it - 1) % self.hp.cosine_annealing["step"] == 0
            if not first_it and it_after:
                self.times += 1
                for g in self.optimizer.param_groups:
                    g["lr"] = self.hp.lr / (2**self.times)
                self.min_eta /= 2
                self.scheduler = CosineAnnealingLR(
                    self.optimizer, self.steps, self.min_eta
                )

    def update_description_bar(self, train_iterator):
        """Update tqdm progress bar with current loss values.

        Parameters
        ----------
        train_iterator : tqdm iterator
        """
        text = f"Iterations [{self.it}/{self.hp.max_iters}]"
        for key in self.hp["losses"].keys():
            if self.hp["losses"][key]["report"]:
                r_loss = self.loss_values[key][-1].item()
                if "log" in self.hp["losses"][key]:
                    if self.hp["losses"][key]["log"]:
                        r_loss = np.log(r_loss + 1e-15)
                text += f" Loss {key.upper()}: {r_loss:.4f}"
        train_iterator.set_description(text)

    def largest_bs(self):
        """Return the largest batch size among all configured losses.

        Returns
        -------
        int
        """
        return max([self.hp["losses"][key]["bs"] for key in self.hp["losses"].keys()])

    def test_loop(self):
        """Run inference on the full test set in batches.

        Returns
        -------
        torch.Tensor
            Concatenated predictions.
        """
        bs = self.largest_bs()
        batch_idx = torch.arange(0, self.n_test, dtype=int, device=self.device)

        predictions = []
        with torch.autocast(
            device_type=self.device, dtype=self.dtype, enabled=self.use_amp
        ):
            for i in self.range(0, self.n_test, bs, leave=False):
                idx = batch_idx[i : (i + bs)]
                pred = self.model(self.test_set.samples[idx])
                predictions.append(pred)
        return torch.cat(predictions)

    def write(self, text):
        """Print text using tqdm.write (if verbose) or plain print.

        Parameters
        ----------
        text : str
        """
        if self.hp.verbose:
            tqdm.write(text)
        else:
            print(text)

    def predict_test(self):
        """Evaluate the model on the test set.

        Returns
        -------
        float
            Test loss value.
        """
        with torch.no_grad():
            predictions = self.test_loop()
            loss_fn = self.loss_fn[self.hp.validation_loss]
            test_loss = loss_fn(
                predictions, self.test_set.targets[: predictions.shape[0]]
            ).item()
        if self.hp.verbose:
            self.write(f"[{self.it}/{self.hp.max_iters}] Test Error: {test_loss:>4f}")
        return test_loss

    def test_and_maybe_save(self, it):
        """Evaluate on test set and save model if it's the best so far.

        Parameters
        ----------
        it : int
            Current iteration number.
        """
        if it == 1:
            if self.hp.save_model:
                torch.save(self.model.state_dict(), self.hp.pth_name)
        elif it % self.hp.test_frequency == 0:
            test_score = self.predict_test()
            if test_score < self.best_test_score:
                self.best_test_score = test_score
                # best_it = it
                if self.hp.verbose:
                    self.write("Best score (just above)")
                if self.hp.save_model:
                    torch.save(self.model.state_dict(), self.hp.pth_name)
            self.test_scores.append(test_score)

    def optuna_stop(self, it):
        """Report to Optuna and raise TrialPruned if pruning is triggered.

        Parameters
        ----------
        it : int
            Current iteration number.
        """
        start = self.hp.optuna["patience"]
        if (
            self.trial
            and it != 1
            and self.it > start
            and it % self.hp.test_frequency == 0
        ):
            self.trial.report(self.test_scores[-1], it)
            if self.trial.should_prune():
                self.hp.best_score = self.best_test_score
                np.savez(
                    self.hp.npz_name,
                    **self.hp,
                )
                raise optuna.exceptions.TrialPruned()

    def convert_last_loss_value(self):
        """Convert the last loss tensor in each loss list to a Python float."""
        for k in self.loss_values.keys():
            self.loss_values[k][-1] = self.loss_values[k][-1].item()

    def early_stop(self, it, loss, break_loop):
        """Check early stopping conditions.

        Parameters
        ----------
        it : int
            Current iteration.
        loss : torch.Tensor
            Current total loss (checked for NaN).
        break_loop : bool
            Incoming break flag.

        Returns
        -------
        bool
            True if training should stop.
        """
        if torch.isnan(loss):
            break_loop = True
            print("Loss has become NaN, stopping...")
        if self.hp.early_stopping["status"]:
            if it != 1 and it % self.hp.test_frequency == 0:
                patience = self.hp.early_stopping["patience"]
                threshold = self.hp.early_stopping["value"]
                ignore_first = self.hp.early_stopping["ignore_first"]
                if len(self.test_scores) + 1 - ignore_first > patience:
                    ref = self.test_scores[-patience]
                    other_scores = np.array(self.test_scores[-patience + 1 :])
                    if (ref + threshold - other_scores < 0).all():
                        break_loop = True
                        print("Early stopping")
        return break_loop

    def autocasting(self):
        """Configure mixed-precision autocasting settings.

        Uses bfloat16 on CPU, float16 on CUDA. Disabled for WIRES
        (complex dtypes incompatible with AMP).
        """
        if self.device == "cpu":
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        self.use_amp = True
        if self.hp.model["name"] == "WIRES":
            self.use_amp = False
        self.dtype = dtype

    def load_best_model(self):
        """Reload the best model checkpoint from disk."""
        self.model.load_state_dict(
            torch.load(self.hp.pth_name, map_location=self.device)
        )

    def setup_validation_loss(self):
        """Ensure the validation loss function exists in self.loss_fn.

        Creates it if the validation loss key wasn't in hp.losses.
        """
        if self.hp.validation_loss not in self.loss_fn:
            if self.hp.validation_loss == "mse":
                self.loss_fn[self.hp.validation_loss] = RMSELoss(
                    ignore_nan=self.hp.ignore_nan
                )
            elif self.hp.validation_loss == "mae":
                self.loss_fn[self.hp.validation_loss] = WL1Loss()

    def fit(self):
        """Run the full training loop.

        Sets up optimiser, scheduler, losses, and iterates for
        hp.max_iters steps. Handles mixed-precision training,
        gradient clipping, loss balancing, early stopping, and
        model checkpointing.
        """
        scaler = torch.amp.GradScaler(self.device, enabled=self.use_amp)
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_losses()
        self.setup_validation_loss()
        self.setup_temporal_causality()
        self.model.train()
        self.best_test_score = np.inf

        iterators = self.range(1, self.hp.max_iters + 1, 1)

        for self.it in iterators:
            self.optimizer.zero_grad(set_to_none=True)

            data_batch = next(self.data)
            with torch.autocast(
                device_type=self.device, dtype=self.dtype, enabled=self.use_amp
            ):
                target_pred = self.model(data_batch["x"])
                self.compute_loss(target_pred, **data_batch)
                self.loss_balancing()
                loss = self.sum_loss(self.loss_values, self.lambdas_scalar)
            scaler.scale(loss).backward()
            self.clip_gradients()

            scaler.step(self.optimizer)
            scale = scaler.get_scale()
            scaler.update()
            skip_lr_sched = scale > scaler.get_scale()
            if self.scheduler_status and not skip_lr_sched:
                self.scheduler_update()
            if self.hp.verbose:
                self.update_description_bar(iterators)

            self.test_and_maybe_save(self.it)
            self.optuna_stop(self.it)
            break_loop = False
            break_loop = self.early_stop(self.it, loss, break_loop)
            if break_loop:
                break
        self.convert_last_loss_value()
        if self.hp.save_model:
            self.load_best_model()
