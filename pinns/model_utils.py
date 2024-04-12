from functools import partial
import torch
import torch.nn as nn
from numpy import sqrt
import torch.jit as jit
import torch._dynamo
from typing import Optional


class Linear(nn.Module):
    def __init__(self, size_in, size_out, is_last=False, act=nn.Tanh):
        super().__init__()
        self.is_last = is_last
        self.layer = torch.nn.Linear(size_in, size_out)
        if not is_last:
            self.layer = nn.Sequential(self.layer, act())
    def forward(self, x):
        return  self.layer(x)

class LinearLayerGlorot(nn.Module):
    """Custom Linear layer but mimics a standard linear layer"""

    def __init__(self, size_in, size_out, is_last=False, act=nn.Tanh):
        super().__init__()
        self.layer = nn.Linear(size_in, size_out, bias=True)
        self.weights = nn.Parameter(
            torch.Tensor(size_out, size_in)
        )  # nn.Parameter is a Tensor that's a module parameter.
        self.bias = nn.Parameter(torch.Tensor(size_out))
        # initialize weights and biases
        nn.init.xavier_normal_(self.weights, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.bias)  # bias init

        self.layer.weight = nn.Parameter(self.weights, requires_grad=True)
        self.layer.bias = nn.Parameter(self.bias, requires_grad=True)
        if is_last:
            self.layer = nn.Sequential(self.layer, act())
    def forward(self, x):
        return  self.layer(x)

class LinearLayerRWF(nn.Module):
    """Custom Linear layer but mimics a standard linear layer"""

    def __init__(self, size_in, size_out, mean, std, is_last=False, act=nn.Tanh):
        super().__init__()
        
        self.layer = torch.nn.Linear(size_in, size_out)
        w = torch.Tensor(size_out, size_in)
        nn.init.kaiming_uniform_(w)  # , gain=nn.init.calculate_gain('relu'))
        s = torch.Tensor(
            size_out,
        )
        nn.init.normal_(s, mean=mean, std=std)
        s = torch.exp(s)

        v = w / s[:, None]

        v_weights = nn.Parameter(v, requires_grad=True)
        s_weights = nn.Parameter(s, requires_grad=True)  
        # nn.Parameter is a Tensor that's a module parameter.
        bias = nn.Parameter(torch.Tensor(size_out))

        nn.init.zeros_(bias)  # bias init
        self.layer.weight = v_weights * s_weights[:, None]
        self.layer.bias = bias
        if is_last:
            self.layer = nn.Sequential(self.layer, act())

    def forward(self, x):
        return self.layer(x)

class RFFLayer(jit.ScriptModule):
    def __init__(
        self,
        input_channel_size,
        mapping_size, #size of the output
        sigma
    ):
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
    return torch.sin(w0 * x)


class SirenLayer(jit.ScriptModule):
    def __init__(
        self,
        in_f,
        out_f,
        is_first=False,
        is_last=False,
        w0=30,
    ):
        super().__init__()

        self.in_f = in_f
        self.w0 = torch.tensor(w0)
        self.linear = nn.Linear(in_f, out_f, bias=True)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / self.in_f if self.is_first else sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)
            self.linear.bias.uniform_(-3, 3)

    @jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return x if self.is_last else sin(x, self.w0)

# @torch.jit.script
def wires(x, omega_0, scale_0):
    omega = omega_0 * x
    scale = scale_0 * x
    return torch.exp(1j*omega - scale.abs().square())

class ComplexGaborLayer(jit.ScriptModule):
    
    def __init__(self, in_f, out_f, is_first=False,
                 is_last=False, omega0=10.0, sigma0=40.0, trainable=True):
        super().__init__()
        self.in_f = in_f
        if is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
        self.is_last = is_last
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(omega0*torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(sigma0*torch.ones(1), trainable)
        
        self.linear = nn.Linear(in_f,
                                out_f,
                                bias=True,
                                dtype=dtype)
    # @jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.is_last:
            return x.real
        else:
            return wires(x, self.omega_0, self.scale_0)

class MFN_Layer(nn.Module): #jit.ScriptModule):
    def __init__(
            self, in_f, out_f, is_first=False,
            is_last=False, in_f0=2, act=nn.Tanh):
        super().__init__()
        self.is_first = is_first
        self.is_last = is_last
        g = nn.Linear(in_f0, out_f, bias=True)
        self.g_layer = nn.Sequential(g, act())
        self.layer = nn.Linear(in_f, out_f, bias=True)
        
    # @jit.script_method
    def forward(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        if self.is_first:
            return self.g_layer(x)
        if self.is_last:
            return self.layer(x)
        return self.layer(x) * self.g_layer(x0)


class MFN(nn.Module):
    def __init__(
            self,
            model,
            hp
        ):
        super().__init__()
        self.model = model
        device = "cuda" if hp.gpu else "cpu"
        self.fake_input = torch.tensor([1], device=device)

    def forward(self, *args):
        x0 = torch.cat(args, axis=1)
        for i, layer in enumerate(self.model):
            if i == 0:
                x = layer(x0, self.fake_input)
            else:
                x = layer(x, x0)
        return x

class SkipLayer(jit.ScriptModule):
    def __init__(
            self,
            layer,
        ):
        super().__init__()
        self.layer = layer
    @jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layer(x)

class ModifiedMLP(nn.Module):
    def __init__(
            self,
            model,
            act,
            hp,
        ):
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

    def forward(self, *args):
        x = torch.cat(args, axis=1)
        for i, layer in enumerate(self.model):
            if i <= self.layer_uv or i == self.n - 1:
                if i == self.layer_uv:
                    Ux = self.U(x)
                    Vx = self.V(x)
                x = layer(x)
            else:
                x = torch.mul(Ux, x) + torch.mul(Vx, 1 - x)
        return x


def linear_fn(text, hp, act):
    if hp.model["name"] == "SIREN":
        return SirenLayer
    elif hp.model["name"] == "WIRES":
        return partial(ComplexGaborLayer, omega0=hp.model["omega0"], sigma0=hp.model["sigma0"], trainable=hp.model["trainable"])
    elif hp.model["name"] == "RFF":
        if text == "HE":
            return partial(Linear, act=act)
        elif text == "Glorot":
            return partial(LinearLayerGlorot, act=act)
        elif text == "RWF":
            return partial(LinearLayerRWF, mean=hp.model["mean"], std=hp.model["std"], act=act)
    elif hp.model["name"] == "MFN":
        return partial(MFN_Layer, in_f0=hp.input_size, act=act)