import torch
import pinns

pi = 3.141592653589793238462643383279502884197


def advection_residue(model, x, t, c=1):
    torch.set_grad_enabled(True)
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)

    du_dx = pinns.gradient(u, x)
    du_dt = pinns.gradient(u, t)

    # the pi is essential as we derive with respect to x_tilde
    # and not x, so chaining rule implies we had the scaling
    # factor between x and x_tilde, which is 3.
    residue = du_dt + c / pi * du_dx
    return residue


def soft_periodicity(model, t):
    x_0 = torch.zeros_like(t)
    residue = model(x_0 - 1, t) - model(x_0 + 1, t)
    return residue


class Advection(pinns.DensityEstimator):
    def pde(self, z, z_hat):
        x = pinns.gen_uniform(self.hp.losses["pde"]["bs"], self.device)

        M = self.M if hasattr(self, "M") else None
        temporal_scheme = self.hp.losses["pde"]["temporal_causality"]

        t = pinns.gen_uniform(
            self.hp.losses["pde"]["bs"],
            self.device,
            start=0,
            end=1,
            temporal_scheme=temporal_scheme,
            M=M,
        )
        residue = advection_residue(self.model, x, t, self.hp.c)
        return residue

    def periodicity(self, z, z_hat):
        M = self.M if hasattr(self, "M") else None
        temporal_scheme = self.hp.losses["periodicity"]["temporal_causality"]

        t = pinns.gen_uniform(
            self.hp.losses["periodicity"]["bs"],
            self.device,
            start=0,
            end=1,
            temporal_scheme=temporal_scheme,
            M=M,
        )
        residue = soft_periodicity(self.model, t)
        return residue
