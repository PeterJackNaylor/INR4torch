import torch
import pinns


def h(string, ope, n):
    if "+" in string:
        if ope == "+":
            string = string.split(ope)[0] + ope + str(int(string[-1]) + n)
        elif ope == "-":
            if string[-1] == f"{n}":
                string = string[:-2]
            else:
                string = string.split(ope)[0] + ope + str(int(string[-1]) - n)
    elif "-" in string:
        if ope == "-":
            string = string.split(ope)[0] + ope + str(int(string[-1]) - n)
        elif ope == "+":
            if string[-1] == f"{n}":
                string = string[:-2]
            else:
                string = string.split(ope)[0] + ope + str(int(string[-1]) + n)
    else:
        string = string + ope + str(n)
    return string


kappa = 1


class ComputeAdvection(object):
    def __init__(
        self,
        model,
        x,
        y,
        t,
        dx,
        dy,
        dt,
        alpha,
        alpha_x=None,
        alpha_y=None,
        alpha_t=None,
        PINNS=True,
    ) -> None:
        torch.set_grad_enabled(True)
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)
        self.model = model
        self.x = x
        self.y = y
        self.t = t
        self.dx = dx
        self.dy = dy
        self.dt = dt

        self.alpha = alpha
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y
        self.alpha_t = alpha_t
        self.PINNS = PINNS
        self.mem_dic = {}

    def compute_gradients(self):
        dpsi_dx, dpsi_dy = self.psi_prime("x", "y", "t")

        q = self.ssh2q("x", "y", "t", dpsi_dx, dpsi_dy)
        dq_dt = self.gradt_q(q)
        J1, J2 = self.advection(q, dpsi_dx, dpsi_dy)
        # import pdb; pdb.set_trace()
        return dq_dt + J1 - J2

    def psi_prime(self, x, y, t):
        if self.PINNS:
            psi = self.m_model(x, y, t)
            dpsi_dx = pinns.gradient(psi, self.x) / self.alpha_x
            dpsi_dy = pinns.gradient(psi, self.y) / self.alpha_y
        else:
            psi_x_plus = self.m_model(h(x, "+", 1), y, t)
            psi_x_minus = self.m_model(h(x, "-", 1), y, t)
            dpsi_dx = (psi_x_plus - psi_x_minus) / (2 * self.dx)

            psi_y_plus = self.m_model(x, h(y, "+", 1), t)
            psi_y_minus = self.m_model(x, h(y, "-", 1), t)
            dpsi_dy = (psi_y_plus - psi_y_minus) / (2 * self.dy)
        return dpsi_dx, dpsi_dy

    def m_model(self, x, y, t):
        # written like (x, y, t) or (x+1, y, t+1)

        if (x, y, t) not in self.mem_dic:
            input_x = self.x
            if x[:2] == "x+":
                input_x = input_x + int(x[2]) * self.dx
            elif x[:2] == "x-":
                input_x = input_x - int(x[2]) * self.dx
            input_y = self.y
            if y[:2] == "y+":
                input_y = input_y + int(y[2]) * self.dy
            elif y[:2] == "y-":
                input_y = input_y - int(y[2]) * self.dy
            input_t = self.t
            if t[:2] == "t+":
                input_t = input_t + int(t[2]) * self.dt
            elif t[:2] == "t-":
                input_t = input_t - int(t[2]) * self.dt
            self.mem_dic[(x, y, t)] = self.alpha * self.model(input_x, input_y, input_t)

        return self.mem_dic[(x, y, t)]

    def comp_laplacian(self, x, y, t):
        psi_x_plus = self.m_model(h(x, "+", 1), y, t)
        psi_x_minus = self.m_model(h(x, "-", 1), y, t)

        psi_y_plus = self.m_model(x, h(y, "+", 1), t)
        psi_y_minus = self.m_model(x, h(y, "-", 1), t)

        psi = self.m_model(x, y, t)

        laplacian_x = (psi_x_plus + psi_x_minus - 2 * psi) / self.dx**2
        laplacian_y = (psi_y_plus + psi_y_minus - 2 * psi) / self.dy**2
        laplacian = laplacian_x + laplacian_y

        return laplacian

    def ssh2q(self, x, y, t, dpsi_dx, dpsi_dy):
        psi = self.m_model(x, y, t)
        if self.PINNS:
            laplacian = (
                pinns.gradient(dpsi_dx, self.x, grad_outputs=None) / self.alpha_x
                + pinns.gradient(dpsi_dy, self.y, grad_outputs=None) / self.alpha_y
            )
        else:
            laplacian = self.comp_laplacian(x, y, t)

        q = laplacian - kappa**2 * psi
        return q

    def compute_q(self, x, y, t):
        laplacian = self.comp_laplacian(x, y, t)
        psi = self.m_model(x, y, t)
        return laplacian - kappa**2 * psi

    def gradt_q(self, q, x="x", y="y", t="t", side="left"):  # can be right
        if self.PINNS:
            dq_dt = pinns.gradient(q, self.t) / self.alpha_t
        else:
            if side == "left":
                q_t_plus = self.compute_q(x, y, h(t, "+", 1))
                dq_dt = (q_t_plus - q) / self.dt
            elif side == "right":
                q_t_minus = self.compute_q(x, y, h(t, "-", 1))
                dq_dt = (q - q_t_minus) / self.dt
        return dq_dt

    def advection(self, q, dpsi_dx, dpsi_dy):
        if self.PINNS:
            dq_dx = pinns.gradient(q, self.x) / self.alpha_x
            dq_dy = pinns.gradient(q, self.y) / self.alpha_y

            dpsi_dy_times_dq_dx = dpsi_dy * dq_dx
            dpsi_dx_times_dq_dy = dpsi_dx * dq_dy
        else:
            zeros = torch.zeros_like(dpsi_dx)
            dpsi_dy_plus = torch.max(dpsi_dy, zeros)
            dpsi_dy_minus = torch.min(dpsi_dy, zeros)

            dpsi_dx_plus = torch.max(dpsi_dx, zeros)
            dpsi_dx_minus = torch.min(dpsi_dx, zeros)

            q_x_plus = self.compute_q("x+1", "y", "t")
            q_x_minus = self.compute_q("x-1", "y", "t")
            dq_dx_plus = (q_x_plus - q) / self.dx
            dq_dx_minus = (q - q_x_minus) / self.dx

            q_y_plus = self.compute_q("x", "y+1", "t")
            q_y_minus = self.compute_q("x", "y-1", "t")
            dq_dy_plus = (q_y_plus - q) / self.dy
            dq_dy_minus = (q - q_y_minus) / self.dy

            dpsi_dy_times_dq_dx = (
                dpsi_dy_plus * dq_dx_minus + dpsi_dy_minus * dq_dx_plus
            )
            dpsi_dx_times_dq_dy = (
                dpsi_dx_plus * dq_dy_minus + dpsi_dx_minus * dq_dy_plus
            )

        return dpsi_dx_times_dq_dy, dpsi_dy_times_dq_dx


def advection_residue(model, x, y, t, hp, pinns_b):

    alpha = hp.nv_targets[0][1]
    alpha_x = hp.nv_samples[0][1]
    alpha_y = hp.nv_samples[1][1]
    alpha_t = hp.nv_samples[2][1]

    if pinns_b:
        dt, dx, dy = None, None, None
    else:
        step_t = hp.losses["pde_advection"]["step_time"] / 3600  # convert to days
        dt = step_t / alpha_t
        step_x = hp.losses["pde_advection"]["step_xy"]
        dx = step_x / alpha_x
        step_y = hp.losses["pde_advection"]["step_xy"]
        dy = step_y / alpha_y

    gradients = ComputeAdvection(
        model,
        x,
        y,
        t,
        dx,
        dy,
        dt,
        alpha,
        alpha_x,
        alpha_y,
        alpha_t,
        pinns_b,
    )
    residue = gradients.compute_gradients()

    # gradients_not = ComputeAdvection(model, lat, lon, t, dlat, dlon, dt, alpha, alpha_lat, alpha_lon, alpha_t, not pinns_b)
    # gradients_not.compute_gradients()
    # import pdb; pdb.set_trace()

    return residue


class QGSurface(pinns.DensityEstimator):
    def pde_advection(self, z, z_hat, weight):
        kname = "pde_advection"
        PINNS_b = self.hp.losses[kname]["grad_method"] == "PINNS"
        x = pinns.gen_uniform(
            self.hp.losses[kname]["bs"], self.device, dtype=self.dtype
        )
        y = pinns.gen_uniform(
            self.hp.losses[kname]["bs"], self.device, dtype=self.dtype
        )
        M = self.M if hasattr(self, "M") else None
        temporal_scheme = self.hp.losses[kname]["temporal_causality"]
        eps = self.hp.losses[kname]["eps_temporal_causality"]

        t = pinns.gen_uniform(
            self.hp.losses[kname]["bs"],
            self.device,
            start=-1 + eps,
            end=1 - eps,
            temporal_scheme=temporal_scheme,
            M=M,
            dtype=self.dtype,
        )

        residue = advection_residue(self.model, x, y, t, self.hp, PINNS_b)

        return residue

    def autocasting(self):
        if self.device == "cpu":
            dtype = torch.bfloat32
        else:
            dtype = torch.float32
        self.use_amp = False
        self.dtype = dtype
