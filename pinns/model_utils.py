from __future__ import annotations

from functools import partial
from typing import Optional, Type, Callable

import torch
import torch.nn as nn
from numpy import sqrt
import torch.jit as jit
import torch._dynamo

from .kan_utils import KAN


class Linear(nn.Module):
    """Standard linear layer with optional activation.

    Parameters
    ----------
    size_in : int
        Number of input features.
    size_out : int
        Number of output features.
    is_last : bool, optional
        Whether this is the last layer (no activation). Default: False.
    act : nn.Module class, optional
        Activation function class. Default: nn.Tanh.
    """

    def __init__(
        self,
        size_in: int,
        size_out: int,
        is_last: bool = False,
        act: Type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.is_last = is_last
        self.layer = torch.nn.Linear(size_in, size_out)
        if not is_last:
            self.layer = nn.Sequential(self.layer, act())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class LinearLayerGlorot(nn.Module):
    """Linear layer with Glorot (Xavier) initialisation.

    Parameters
    ----------
    size_in : int
        Number of input features.
    size_out : int
        Number of output features.
    is_last : bool, optional
        Whether this is the last layer (no activation). Default: False.
    act : nn.Module class, optional
        Activation function class. Default: nn.Tanh.
    """

    def __init__(
        self,
        size_in: int,
        size_out: int,
        is_last: bool = False,
        act: Type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.layer = nn.Linear(size_in, size_out, bias=True)
        self.weights = nn.Parameter(
            torch.empty(size_out, size_in)
        )  # nn.Parameter is a Tensor that's a module parameter.
        self.bias = nn.Parameter(torch.empty(size_out))
        # initialize weights and biases
        nn.init.xavier_normal_(self.weights, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.bias)  # bias init

        self.layer.weight = nn.Parameter(self.weights, requires_grad=True)
        self.layer.bias = nn.Parameter(self.bias, requires_grad=True)
        if not is_last:
            self.layer = nn.Sequential(self.layer, act())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class LinearLayerRWF(nn.Module):
    """Linear layer with Random Weight Factorisation (RWF).

    Factorises weights as W = v * s, where s is drawn from a
    log-normal distribution, enabling better spectral control.

    Parameters
    ----------
    size_in : int
        Number of input features.
    size_out : int
        Number of output features.
    mean : float
        Mean of the normal distribution for log-scale factors.
    std : float
        Standard deviation of the normal distribution for log-scale factors.
    is_last : bool, optional
        Whether this is the last layer (no activation). Default: False.
    act : nn.Module class, optional
        Activation function class. Default: nn.Tanh.
    """

    def __init__(
        self,
        size_in: int,
        size_out: int,
        mean: float,
        std: float,
        is_last: bool = False,
        act: Type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()

        self.layer = torch.nn.Linear(size_in, size_out)
        w = torch.empty(size_out, size_in)
        nn.init.kaiming_uniform_(w)  # , gain=nn.init.calculate_gain('relu'))
        s = torch.empty(
            size_out,
        )
        nn.init.normal_(s, mean=mean, std=std)
        s = torch.exp(s)

        v = w / s[:, None]

        v_weights = nn.Parameter(v, requires_grad=True)
        s_weights = nn.Parameter(s, requires_grad=True)
        # nn.Parameter is a Tensor that's a module parameter.
        bias = nn.Parameter(torch.empty(size_out))

        nn.init.zeros_(bias)  # bias init
        self.layer.weight = v_weights * s_weights[:, None]
        self.layer.bias = bias
        if not is_last:
            self.layer = nn.Sequential(self.layer, act())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class RFFLayer(jit.ScriptModule):
    """Random Fourier Features input encoding layer.

    Projects inputs into a higher-dimensional space using random
    Fourier features: [sin(2pi*Bx), cos(2pi*Bx)] where B is a
    fixed random matrix.

    Parameters
    ----------
    input_channel_size : int
        Number of input channels.
    mapping_size : int
        Total size of the output (half sin, half cos).
    sigma : float
        Standard deviation of the random projection matrix B.

    References
    ----------
    Tancik, M. et al. "Fourier Features Let Networks Learn High
    Frequency Functions in Low Dimensional Domains." NeurIPS 2020.
    """

    def __init__(
        self, input_channel_size: int, mapping_size: int, sigma: float
    ) -> None:
        super(RFFLayer, self).__init__()
        self.layer = nn.Linear(input_channel_size, mapping_size // 2, bias=False)
        B = torch.normal(0, sigma, (mapping_size // 2, input_channel_size))
        self.layer.weight = nn.Parameter(B, requires_grad=False)
        self.layer.requires_grad_(False)

    @jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = 2 * torch.pi * self.layer(x)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


@torch.jit.script
def sin(x, w0: int):
    """Scaled sine activation: sin(w0 * x).

    JIT-compiled helper used by SirenLayer.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    w0 : int
        Frequency scaling factor.

    Returns
    -------
    torch.Tensor
        sin(w0 * x).
    """
    return torch.sin(w0 * x)


class SirenLayer(jit.ScriptModule):
    """SIREN layer with sinusoidal activation.

    Implements a linear layer followed by a sine activation,
    with weight initialisation from Sitzmann et al. (2020).

    Parameters
    ----------
    in_f : int
        Number of input features.
    out_f : int
        Number of output features.
    is_first : bool, optional
        Whether this is the first layer (uses different init). Default: False.
    is_last : bool, optional
        Whether this is the last layer (no activation). Default: False.
    w0 : int, optional
        Frequency scaling factor for sine activation. Default: 30.

    References
    ----------
    Sitzmann, V. et al. "Implicit Neural Representations with Periodic
    Activation Functions." NeurIPS 2020.
    """

    def __init__(
        self,
        in_f: int,
        out_f: int,
        is_first: bool = False,
        is_last: bool = False,
        w0: int = 30,
    ) -> None:
        super().__init__()

        self.in_f = in_f
        self.w0 = torch.tensor(w0)
        self.linear = nn.Linear(in_f, out_f, bias=True)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self) -> None:
        b = 1 / self.in_f if self.is_first else sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)
            self.linear.bias.uniform_(-3, 3)

    @jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return x if self.is_last else sin(x, self.w0)


# @torch.jit.script
def wires(
    x: torch.Tensor, omega_0: torch.Tensor, scale_0: torch.Tensor
) -> torch.Tensor:
    """Complex Gabor wavelet activation function.

    Computes exp(i * omega_0 * x - |scale_0 * x|^2).

    Parameters
    ----------
    x : torch.Tensor
        Complex-valued input.
    omega_0 : torch.Tensor
        Frequency parameter (scalar).
    scale_0 : torch.Tensor
        Envelope width parameter (scalar).

    Returns
    -------
    torch.Tensor
        Complex-valued activation output.
    """
    omega = omega_0 * x
    scale = scale_0 * x
    return torch.exp(1j * omega - scale.abs().square())


class ComplexGaborLayer(jit.ScriptModule):
    """Complex Gabor wavelet layer for WIRE networks.

    Applies a complex linear transformation followed by a Gabor
    wavelet activation: exp(i*omega0*x - |sigma0*x|^2).

    Parameters
    ----------
    in_f : int
        Number of input features.
    out_f : int
        Number of output features.
    is_first : bool, optional
        If True, uses real-valued dtype. Default: False.
    is_last : bool, optional
        If True, returns real part only. Default: False.
    omega0 : float, optional
        Frequency parameter. Default: 10.0.
    sigma0 : float, optional
        Envelope width parameter. Default: 40.0.
    trainable : bool, optional
        Whether omega0 and sigma0 are trainable. Default: True.

    References
    ----------
    Saragadam, V. et al. "WIRE: Wavelet Implicit Neural
    Representations." CVPR 2023.
    """

    def __init__(
        self,
        in_f: int,
        out_f: int,
        is_first: bool = False,
        is_last: bool = False,
        omega0: float = 10.0,
        sigma0: float = 40.0,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.in_f = in_f
        if is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
        self.is_last = is_last

        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(omega0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(sigma0 * torch.ones(1), trainable)

        self.linear = nn.Linear(in_f, out_f, bias=True, dtype=dtype)

    # @jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.is_last:
            return x.real
        else:
            return wires(x, self.omega_0, self.scale_0)


class MFN_Layer(nn.Module):
    """Single layer of a Multiplicative Filter Network.

    Each hidden layer multiplies its linear output with a learnable
    filter applied to the original input.

    Parameters
    ----------
    in_f : int
        Number of input features from previous layer.
    out_f : int
        Number of output features.
    is_first : bool, optional
        If True, only applies the filter (no linear branch). Default: False.
    is_last : bool, optional
        If True, only applies the linear layer (no filter). Default: False.
    in_f0 : int, optional
        Number of features of the original input (for the filter). Default: 2.
    act : nn.Module class, optional
        Activation function class for the filter. Default: nn.Tanh.

    References
    ----------
    Fathony, R. et al. "Multiplicative Filter Networks." ICLR 2021.
    """

    def __init__(
        self,
        in_f: int,
        out_f: int,
        is_first: bool = False,
        is_last: bool = False,
        in_f0: int = 2,
        act: Type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.is_first = is_first
        self.is_last = is_last
        g = nn.Linear(in_f0, out_f, bias=True)
        self.g_layer = nn.Sequential(g, act())
        self.layer = nn.Linear(in_f, out_f, bias=True)

    # @jit.script_method
    def forward(
        self, x: torch.Tensor, x0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.is_first:
            return self.g_layer(x)
        if self.is_last:
            return self.layer(x)
        return self.layer(x) * self.g_layer(x0)


class MFN(nn.Module):
    """Multiplicative Filter Network.

    Implements a network where each hidden layer output is modulated
    by a learnable filter applied to the original input.

    Parameters
    ----------
    model : nn.Sequential
        Sequence of MFN_Layer modules.
    hp : AttrDict
        Hyperparameters.

    References
    ----------
    Fathony, R. et al. "Multiplicative Filter Networks." ICLR 2021.
    """

    def __init__(self, model: nn.Sequential, hp: object) -> None:
        super().__init__()
        self.model = model

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        x0 = torch.cat(args, dim=1)
        for i, layer in enumerate(self.model):
            if i == 0:
                x = layer(x0)
            else:
                x = layer(x, x0)
        return x


class SkipLayer(jit.ScriptModule):
    """Residual skip connection wrapper.

    Wraps a layer to produce output = x + layer(x).

    Parameters
    ----------
    layer : nn.Module
        The layer to wrap with a skip connection.
    """

    def __init__(
        self,
        layer: nn.Module,
    ) -> None:
        super().__init__()
        self.layer = layer

    @jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layer(x)


class ModifiedMLP(nn.Module):
    """Modified MLP with U/V encoding branches.

    Implements the modified MLP architecture from Wang et al. (2021)
    where two auxiliary encoding branches U and V modulate the
    intermediate hidden representations.

    Parameters
    ----------
    model : nn.Sequential
        The base MLP model.
    act : nn.Module class
        Activation function class (e.g., nn.Tanh).
    hp : AttrDict
        Hyperparameters containing model configuration.

    References
    ----------
    Wang, S. et al. "Understanding and Mitigating Gradient Flow
    Pathologies in Physics-Informed Neural Networks." SIAM J. Sci. Comput. 2021.
    """

    def __init__(
        self,
        model: nn.Sequential,
        act: Type[nn.Module],
        hp: object,
    ) -> None:
        super().__init__()
        self.hp = hp
        self.model = model
        self.n = len(self.model)
        linear_layer_fn = linear_fn(self.hp.model["linear"], self.hp, act)
        if hp.model["name"] in ["RFF"]:
            input_size = next(self.model[1].layer.parameters()).shape[0]
            self.layer_uv = 1
        elif hp.model["name"] in ["SIREN", "WIRES"]:
            input_size = self.model[1].in_f
            self.layer_uv = 1
        else:
            input_size = hp.input_size
            self.layer_uv = 0
        self.U = linear_layer_fn(input_size, self.hp.model["hidden_width"])
        self.V = linear_layer_fn(input_size, self.hp.model["hidden_width"])

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        x = torch.cat(args, dim=1)
        for i, layer in enumerate(self.model):
            if i <= self.layer_uv or i == self.n - 1:
                if i == self.layer_uv:
                    Ux = self.U(x)
                    Vx = self.V(x)
                x = layer(x)
            else:
                x = torch.mul(Ux, x) + torch.mul(Vx, 1 - x)
        return x


def linear_fn(text: str, hp: object, act: Optional[Type[nn.Module]]) -> Callable:
    """Factory function returning the appropriate layer constructor.

    Parameters
    ----------
    text : str
        Linear layer type: 'HE', 'Glorot', or 'RWF'.
    hp : AttrDict
        Hyperparameters with model configuration.
    act : nn.Module class or None
        Activation function class.

    Returns
    -------
    callable
        A partially-applied layer constructor.
    """
    if hp.model["name"] == "SIREN":
        return SirenLayer
    elif hp.model["name"] == "WIRES":
        return partial(
            ComplexGaborLayer,
            omega0=hp.model["omega0"],
            sigma0=hp.model["sigma0"],
            trainable=hp.model["trainable"],
        )
    elif hp.model["name"] == "RFF":
        if text == "HE":
            return partial(Linear, act=act)
        elif text == "Glorot":
            return partial(LinearLayerGlorot, act=act)
        elif text == "RWF":
            return partial(
                LinearLayerRWF, mean=hp.model["mean"], std=hp.model["std"], act=act
            )
    elif hp.model["name"] == "MFN":
        return partial(MFN_Layer, in_f0=hp.input_size, act=act)
    else:
        raise ValueError(f"Unknown model name: {hp.model['name']}")
