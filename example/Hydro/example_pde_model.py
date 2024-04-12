import torch
import pinns
import random 

pi = 3.141592653589793238462643383279502884197


def conservation_residue(model, t, nv_samples, nv_targets, var_pos):
    torch.set_grad_enabled(True)
    t.requires_grad_(True)
    u = model(t)

    P = u[:, 0]
    idx_p = var_pos["P"]
    P = P * nv_targets[idx_p][1] + nv_targets[idx_p][0]

    idx_e = var_pos["E"][0]
    E = u[:, 1]
    E = E * nv_targets[idx_e][1] + nv_targets[idx_e][0]

    idx_ds = var_pos["DS"][0]
    delta_s = u[:,2]
    ddeltaS_dt = (nv_targets[idx_ds][1] / nv_samples[0][1]) * pinns.gradient(delta_s, t)[:, 0] / 30.4

    idx_q = var_pos["Q"]
    Q = u[:, 3]
    Q = Q * nv_targets[idx_q][1] + nv_targets[idx_q][0]

    residue = ddeltaS_dt - P + E + Q

    return residue 


def spatial_grad(model, x, t):
    torch.set_grad_enabled(True)
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)
    du_dx = pinns.gradient(u, x)
    return du_dx



class Hydro(pinns.DensityEstimator):
    def __init__(self, train, test, model, model_hp, gpu, trial=None):
        self.data = train
        self.test_set = test
        self.n = len(train)
        self.n_test = len(test)
        # because of the multiple E and DS
        self.n_outputs = self.data.output_size

        self.model = model
        self.hp = model_hp
        self.device = "cuda" if gpu else "cpu"
        self.trial = trial
        self.autocasting()

    def pde(self, z, z_hat, weight):
        # conservation law
        M = self.M if hasattr(self, "M") else None
        temporal_scheme = self.hp.losses["pde"]["temporal_causality"]

        t = pinns.gen_uniform(
            self.hp.losses["pde"]["bs"],
            self.device,
            start=-1,
            end=1,
            temporal_scheme=temporal_scheme,
            M=M,
            dtype=self.dtype,
        )
        residue = conservation_residue(self.model, t, self.hp.nv_samples, self.hp.nv_targets, self.hp.var_pos)
        return residue

    def spatial_gradient(self, z, z_hat, weight):
        x = pinns.gen_uniform(self.hp.losses["spatial_grad"]["bs"], self.device)

        M = self.M if hasattr(self, "M") else None
        temporal_scheme = self.hp.losses["spatial_grad"]["temporal_causality"]

        t = pinns.gen_uniform(
            self.hp.losses["spatial_grad"]["bs"],
            self.device,
            start=0,
            end=1,
            temporal_scheme=temporal_scheme,
            M=M,
            dtype=self.dtype,
        )
        grad = spatial_grad(self.model, x, t)
        return grad

    # def predict_test(self):
    #     predictions = self.test_loop()
    #     loss_fn = self.loss_fn[self.hp.validation_loss]
    #     test_set = self.test_set
    #     with torch.autocast(
    #         device_type=self.device, dtype=self.dtype, enabled=self.use_amp
    #     ):
    #         # test_set.column_index[test_set.E_pos] = random.choice(test_set.hp.var_pos["E"])
    #         # test_set.column_index[test_set.DS_pos] = random.choice(test_set.hp.var_pos["DS"])
    #         test_loss = loss_fn(
    #             predictions, test_set.targets[: predictions.shape[0], test_set.column_index]
    #         ).item()
    #     if self.hp.verbose:
    #         self.write(f"[{self.it}/{self.hp.max_iters}] Test Error: {test_loss:>4f}")
    #     return test_loss

    # def test_loop(self):
    #     column_index = self.data.column_index
    #     E_pos = self.data.E_pos
    #     DS_pos = self.data.DS_pos
    #     bs = self.largest_bs()
    #     batch_idx = torch.arange(0, self.n_test, dtype=int, device=self.device)
        
    #     predictions = []
    #     with torch.autocast(
    #         device_type=self.device, dtype=self.dtype, enabled=self.use_amp
    #     ):
    #         with torch.no_grad():
    #             for i in self.range(0, self.n_test, bs, leave=False):
    #                 column_index[E_pos] = random.choice(self.hp.var_pos["E"])
    #                 column_index[DS_pos] = random.choice(self.hp.var_pos["DS"])
    #                 samples = self.test_set.samples[:, column_index]
    #                 idx = batch_idx[i : (i + bs)]
    #                 pred = self.model(samples[idx])
    #                 predictions.append(pred)
    #     return torch.cat(predictions)
