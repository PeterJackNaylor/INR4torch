import math
import copy
import numpy as np
from tqdm import trange, tqdm
from scipy.special import softmax
import optuna
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import collections


def skip_first(it):
    """
    Skip the first element of an Iterator or Iterable,
    like a Generator or a list.
    This will always return a generator or raise TypeError()
    in case the argument's type is not compatible
    """
    if isinstance(it, collections.Iterator):
        try:
            next(it)
            yield from it
        except StopIteration:
            return
    elif isinstance(it, collections.Iterable):
        yield from skip_first(it.__iter__())
    else:
        raise TypeError(
            f"You must pass an Iterator or an Iterable to skip_first(), but you passed {it}"
        )


def Norm1(vector, dim=None):
    return torch.norm(vector, p=1, dim=dim)


def Norm2(vector, dim=None):
    return torch.norm(vector, p=2, dim=dim)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


def sum_loss(dic_values, dic_lambdas):
    loss = 0.0
    for key in dic_lambdas.keys():
        loss += dic_lambdas[key][-1] * dic_values[key][-1]

    if len(dic_values[list(dic_values.keys())[0]]) > 1:
        for key in dic_lambdas.keys():
            dic_values[key][-2] = dic_values[key][-2].detach().item()
    return loss


def checkpoint_values(dic_values, dic_tmp_values):
    for key in dic_values.keys():
        dic_values[key].append(torch.stack(dic_tmp_values[key]).mean().item())


def model_params_data(model, hp):
    params = model.parameters()
    for p in params:
        try:
            data = p.grad.data
            yield data
        except:
            pass
    # if hp.model["name"] == "RFF":
    #     params = skip_first(params)
    # return params


def balancing_loss(loss_values, lambdas, alpha, model, hp, device):
    grad_norms = {}
    for k in loss_values.keys():
        if lambdas[k][-1] != 0:
            loss_k = loss_values[k][-1]
            loss_k.backward(retain_graph=True)
            grad_loss_k = [
                p.flatten().clone() for p in model_params_data(model, hp)
            ]
            grad_norms[k] = Norm2(torch.cat(grad_loss_k))
            if device != "cpu":
                grad_norms[k] = grad_norms[k].cpu()
            grad_norms[k] = grad_norms[k]
            for p in model_params_data(model, hp):
                p.zero_()
        else:
            grad_norms[k] = 0
    summed_grads = np.sum(list(grad_norms.values()))
    for k in loss_values.keys():
        if lambdas[k][-1] != 0:
            new_val = (summed_grads / grad_norms[k]).item()
            new_val = lambdas[k][-1] * alpha + new_val * (1 - alpha)
        else:
            new_val = 0
        lambdas[k].append(new_val)


class DensityEstimator:
    def __init__(self, train, test, model, model_hp, gpu, trial=None):

        self.data = train
        self.test_set = test
        self.n = len(train)
        self.n_test = len(test)

        self.model = model
        self.hp = model_hp
        self.device = "cuda" if gpu else "cpu"
        self.trial = trial

    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hp.lr,
            eps=self.hp.eps,
        )

    def setup_scheduler(self):
        scheduler_status = True
        if self.hp.learning_rate_decay["status"]:
            step_size = self.hp.learning_rate_decay["step"]
            scheduler = StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=self.hp.learning_rate_decay["gamma"],
            )
            scheduler_status = True
        elif self.hp.cosine_anealing["status"]:
            self.steps = self.hp.cosine_anealing["step"]
            self.min_eta = self.hp.cosine_anealing["min_eta"]
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
        lambdas = {}
        loss_values = {}
        loss_fn = {}

        for key in self.hp.losses.keys():

            if "lambda" in self.hp.losses[key]:
                lambdas[key] = [self.hp.losses[key]["lambda"]]
            else:
                lambdas[key] = [1]
            if key == "mse":
                loss_fn[key] = RMSELoss()  # RMSELoss()
            elif key == "mae":
                loss_fn[key] = nn.L1Loss()
            else:
                if "method" in self.hp.losses[key]:
                    loss_computation_fn = getattr(self, self.hp.losses[key]["method"])
                    bs_loss = self.hp.losses[key]["bs"]
                    if self.hp.losses[key]["temporal_causality"]:
                        loss_fn[key] = self.temporal_loss(
                            key, loss_computation_fn, bs_loss
                        )
                    else:
                        loss_fn[key] = self.standard_loss(
                            key, loss_computation_fn, bs_loss
                        )
            loss_values[key] = []
        self.lambdas_scalar = lambdas
        self.loss_values = loss_values
        self.loss_fn = loss_fn
        self.test_scores = []

    def setup_temporal_causality(self):
        self.temporal_weights = {}
        for key in self.hp["losses"].keys():
            if "temporal_causality" in self.hp["losses"][key]:
                if self.hp["losses"][key]["temporal_causality"]:
                    M = self.hp.temporal_causality["M"]
                    self.temporal_weights[key] = [torch.ones(M, requires_grad=False, device=self.device)]
                    self.M = M
                    self.eps = self.hp.temporal_causality["eps"]

    def temporal_loss(self, key, residue_computation, bs_loss):
        def f(z, zhat):
            residue = residue_computation(z, zhat)
            square_res = residue.reshape(bs_loss // self.M, self.M)
            M_losses = Norm2(square_res, dim=0)
            shifted_M_loss = torch.zeros_like(M_losses)

            shifted_M_loss[1:] = M_losses[:-1]
            w_i = torch.exp(-self.eps * torch.cumsum(shifted_M_loss, dim=0)).detach()
            w_i.requires_grad_(False)
            self.temporal_weights[key].append(w_i)
            loss = torch.mean(w_i * M_losses)
            return loss

        return f

    def compute_loss(self, z, zhat):
        for key in self.hp.losses.keys():
            self.loss_values[key].append(self.loss_fn[key](z, zhat))

    def standard_loss(self, key, residue_computation, bs_loss):
        def f(z, zhat):
            residue = residue_computation(z, zhat).flatten()
            loss = Norm2(residue) / (residue.shape[0]) ** 0.5
            return loss
        return f

    def range(self, start, end, step, leave=True):
        iterator_fn = trange if self.hp.verbose else range
        if not leave and self.hp.verbose:
            return iterator_fn(start, end, step, leave=leave)
        return iterator_fn(start, end, step)

    def loss_balancing(self):
        if self.hp.self_adapting_loss_balancing["status"]:
            f = self.hp.self_adapting_loss_balancing["step"]
            # iter_b = self.hp.self_adapting_loss_balancing["i"]
            if self.it != 1 and self.it % f == 0:
                alpha = self.hp.self_adapting_loss_balancing["alpha"]
                balancing_loss(
                    self.loss_values,
                    self.lambdas_scalar,
                    alpha,
                    self.model,
                    self.hp,
                    self.device,
                )
        elif self.hp.relobralo["status"]:
            f = self.hp.relobralo["step"]
            if self.it != 1 and self.it % f == 0:
                rho = self.hp.relobralo["rho"]
                alpha = self.hp.relobralo["alpha"]
                T = self.hp.relobralo["T"]
                with torch.no_grad():
                    lambs_hat = softmax(
                        [
                            self.loss_values[i][-1]
                            / (self.loss_values[i][-2] * T + 1e-12)
                            for i in self.loss_values.keys()
                        ]
                    ) * len(self.loss_values.keys())
                    lambs0_hat = softmax(
                        [
                            self.loss_values[i][-1]
                            / (self.loss_values[i][0] * T + 1e-12)
                            for i in self.loss_values.keys()
                        ]
                    ) * len(self.loss_values.keys())
                    for i, key in enumerate(self.loss_values.keys()):
                        self.lambdas_scalar[key].append(
                            rho * alpha * self.lambdas_scalar[key][-1]
                            + (1 - rho) * alpha * lambs0_hat[i]
                            + (1 - alpha) * lambs_hat[i]
                        )

    def clip_gradients(self):
        if self.hp.clip_gradients:
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

    def scheduler_update(self):
        self.lr_list.append(self.scheduler.get_last_lr()[0])
        self.scheduler.step()

        if self.hp.cosine_anealing["status"]:
            # so that is starts on the end of the it number, or
            # on the start of the new one
            # if self.it == 21:
            #     import pdb; pdb.set_trace()
            first_it = self.it == 1
            it_after = (self.it - 1) % self.hp.cosine_anealing["step"] == 0
            if not first_it and it_after:
                self.times += 1
                for g in self.optimizer.param_groups:
                    g["lr"] = self.hp.lr / (2**self.times)
                self.min_eta /= 2
                self.scheduler = CosineAnnealingLR(
                    self.optimizer, self.steps, self.min_eta
                )

    def update_description_bar(self, train_iterator):
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
        return max([self.hp["losses"][key]["bs"] for key in self.hp["losses"].keys()])

    def test_loop(self):
        bs = self.largest_bs()
        batch_idx = torch.arange(0, self.n_test, dtype=int, device=self.device)

        predictions = []
        with torch.no_grad():
            for i in self.range(0, self.n_test, bs, leave=False):
                idx = batch_idx[i : (i + bs)]
                pred = self.model(self.test_set.samples[idx])
                predictions.append(pred)
        return torch.cat(predictions)

    def write(self, text):
        if self.hp.verbose:
            tqdm.write(text)
        else:
            print(text)

    def predict_test(self):
        predictions = self.test_loop()
        loss_fn = self.loss_fn[self.hp.validation_loss]
        test_loss = loss_fn(
            predictions, self.test_set.targets[: predictions.shape[0]]
        ).item()
        if self.hp.verbose:
            self.write(f"[{self.it}/{self.hp.max_iters}] Test Error: {test_loss:>4f}")
        return test_loss

    def test_and_maybe_save(self, it):
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
        if self.trial and it != 1 and it % self.hp.test_frequency == 0:
            self.trial.report(self.test_scores[-1], it)
            if self.trial.should_prune():
                if "B" in self.hp.keys():
                    self.hp.B = np.array(self.hp.B.cpu())
                self.hp.best_score = self.best_test_score
                np.savez(
                    self.hp.npz_name,
                    **self.hp,
                )
                raise optuna.exceptions.TrialPruned()


    def convert_last_loss_value(self):
        for k in self.loss_values.keys():
            self.loss_values[k][-1] = self.loss_values[k][-1].item()


    def early_stop(self, it, loss, break_loop):
        if torch.isnan(loss):
            break_loop = True
        if self.hp.early_stopping["status"]:
            if it != 1 and it % self.hp.test_frequency == 0:
                patience = self.hp.early_stopping["patience"]
                threshold = self.hp.early_stopping["value"]
                if len(self.test_scores) + 1 > patience:
                    ref = self.test_scores[-patience]
                    other_scores = np.array(self.test_scores[-patience+1:])
                    if (ref + threshold - other_scores < 0).all():
                        break_loop = True
        return break_loop

    def fit(self):
        use_amp = True
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_losses()
        self.setup_temporal_causality()
        self.model.train()
        self.best_test_score = np.inf
        # best_it = 0
        iterators = self.range(1, self.hp.max_iters + 1, 1)
        for self.it in iterators:
            self.optimizer.zero_grad()

            data_batch = next(self.data)
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=use_amp):
                target_pred = self.model(data_batch[0])
                true_pred = data_batch[1]

                self.compute_loss(true_pred, target_pred)
                self.loss_balancing()
                loss = sum_loss(self.loss_values, self.lambdas_scalar)
            scaler.scale(loss).backward()
            # loss.backward()
            self.clip_gradients()

            # self.optimizer.step()
            scaler.step(self.optimizer)
            scaler.update()
            if self.scheduler_status:
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
