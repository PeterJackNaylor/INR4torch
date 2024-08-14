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

    def autocasting(self):
        if self.device == "cpu":
            dtype = torch.bfloat16
            if self.hp.model["name"] == "WIRES":
                dtype = torch.bfloat32
        else:
            dtype = torch.float16
            if self.hp.model["name"] == "WIRES":
                dtype = torch.float32
        self.use_amp = True
        if self.hp.model["name"] == "WIRES":
            self.use_amp = False
        self.dtype = dtype


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


def conservation_residue_forecastcombi(model, t, nv_samples):
    torch.set_grad_enabled(True)
    t.requires_grad_(True)
    u = model(t)
    PEDSQ = u[1]
    P = PEDSQ[:, 0]
    E = PEDSQ[:, 1]
    delta_s = PEDSQ[:, 2]
    ddeltaS_dt = (1 / nv_samples[0][1]) * pinns.gradient(delta_s, t)[:, 0] / 30.4

    Q = PEDSQ[:, 3]
    residue = ddeltaS_dt - P + E + Q

    return residue 

def conditional_conservation_residue_forecastcombi(model, t, x, nv_samples):
    torch.set_grad_enabled(True)
    t.requires_grad_(True)
    u = model(t, x)
    PEDSQ = u[1]
    P = PEDSQ[:, 0]
    E = PEDSQ[:, 1]
    delta_s = PEDSQ[:, 2]
    ddeltaS_dt = (1 / nv_samples[0][1]) * pinns.gradient(delta_s, t)[:, 0] / 30.4

    Q = PEDSQ[:, 3]
    residue = ddeltaS_dt - P + E + Q

    return residue 
class HydroForecastCombi(Hydro):
    def compute_loss(self, zhat, **args_dic):
        for key in self.hp.losses.keys():
            if key != "pde":
                self.loss_values[key].append(self.loss_fn[key](zhat[0], **args_dic))
            else:
                self.loss_values[key].append(self.loss_fn[key](zhat[1], **args_dic))

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
        residue = conservation_residue_forecastcombi(self.model, t, self.hp.nv_samples)
        return residue
    def test_loop(self):
        bs = self.largest_bs()
        batch_idx = torch.arange(0, self.n_test, dtype=int, device=self.device)

        predictions = []
        with torch.autocast(
            device_type=self.device, dtype=self.dtype, enabled=self.use_amp
        ):
            for i in self.range(0, self.n_test, bs, leave=False):
                idx = batch_idx[i : (i + bs)]
                pred = self.model(self.test_set.samples[idx])[0]
                predictions.append(pred)
        return torch.cat(predictions)
    

    def test_loop_real(self):
        bs = self.largest_bs()
        batch_idx = torch.arange(0, self.n_test, dtype=int, device=self.device)

        predictions = []
        with torch.autocast(
            device_type=self.device, dtype=self.dtype, enabled=self.use_amp
        ):
            for i in self.range(0, self.n_test, bs, leave=False):
                idx = batch_idx[i : (i + bs)]
                pred = self.model(self.test_set.samples[idx])[1]
                predictions.append(pred)
        return torch.cat(predictions)
    
class CombiHydroForecastCombi(HydroForecastCombi):
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
        x = torch.randint(0, 6, (self.hp.losses["pde"]["bs"], 1), device=self.device, dtype=self.dtype)
        residue = conditional_conservation_residue_forecastcombi(self.model, t, x, self.hp.nv_samples)
        return residue

# class LSTMCombiHydroForecastCombi(CombiHydroForecastCombi):
