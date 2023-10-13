import torch
import torch.nn as nn
from numpy import random, float32, sqrt
import torch.nn.functional as F


def gen_b(mapping, scale, input_size, gpu=False):
    shape = (mapping, input_size)
    B = torch.tensor(random.normal(size=shape).astype(float32))
    if gpu:
        B = B.to("cuda")
    return B * scale


def ReturnModel(
    input_size,
    output_size=1,
    hp=True,  # actually simply a dictionnary
):
    if hp.model["name"] == "RFF":
        mod = RFF(input_size, output_size, hp)
    elif hp.model["name"] == "SIREN":
        mod = SIREN(
            input_size,
            output_size,
            hp,
        )

    return mod


class RFF(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hp,
    ):

        super(RFF, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hp = hp

        self.setup()

        self.gen_architecture()

        if hp.normalise_targets:
            self.final_act = torch.tanh
        else:
            self.final_act = nn.Identity()

    def setup(self):
        if self.hp.model["activation"] == "tanh":
            self.act = nn.Tanh
        elif self.hp.model["activation"] == "relu":
            self.act = nn.ReLU
        self.fourier_mapping_setup(self.hp.B)
        self.first = self.fourier_map
        self.input_size = self.fourier_size * 2

    def fourier_mapping_setup(self, B):
        n, p = B.shape
        layer = nn.Linear(n, p, bias=False)
        layer.weight = nn.Parameter(B, requires_grad=False)
        layer.requires_grad_(False)
        self.fourier_layer = layer
        self.fourier_size = n

    def fourier_map(self, x):
        x = 2 * torch.pi * self.fourier_layer(x)
        x = torch.cat([torch.sin(x), torch.cos(x)], axis=-1)
        return x

    def gen_architecture(self):
        layers = []
        for i in range(self.hp.model["hidden_nlayers"]):
            if i == 0:
                width_i = self.input_size
            else:
                width_i = self.hp.model["hidden_width"]

            layer = nn.Sequential(
                nn.Linear(width_i, self.hp.model["hidden_width"]), self.act()
            )
            layers.append(layer)

        layer = nn.Sequential(
            nn.Linear(self.hp.model["hidden_width"], self.output_size),
        )
        layers.append(layer)

        self.mlp = nn.Sequential(*layers)

    def forward(self, *args):
        xin = torch.cat(args, axis=1)
        xin = self.first(xin)
        if self.hp.model["skip"]:
            for i, layer in enumerate(self.mlp.model):
                if layer.is_first:
                    x = layer(xin)
                elif layer.is_last:
                    out = layer(x)
                else:
                    x = layer(x) + x if i % 2 == 1 else layer(x)
        else:
            out = self.mlp(xin)
        # out = torch.squeeze(out)
        return self.final_act(out)


# create the GON network (a SIREN as in https://vsitzmann.github.io/siren/)
class SirenLayer(nn.Module):
    def __init__(
        self,
        in_f,
        out_f,
        w0=30,
        is_first=False,
        is_last=False,
    ):
        super().__init__()

        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / self.in_f if self.is_first else sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)
            self.linear.bias.uniform_(-3, 3)

    def forward(self, x, cond_freq=None, cond_phase=None):
        x = self.linear(x)
        if cond_freq is not None:
            freq = cond_freq  # .unsqueeze(1).expand_as(x)
            x = freq * x
        if cond_phase is not None:
            phase_shift = cond_phase  # unsqueeze(1).expand_as(x)
            x = x + phase_shift
        return x if self.is_last else torch.sin(self.w0 * x)


class SIREN_model(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_layers,
        width_layers,
        features_scales,
    ):
        super(SIREN_model, self).__init__()
        self.hidden_width = width_layers
        model_dim = [input_size] + hidden_layers * [width_layers] + [output_size]

        first_layer = SirenLayer(
            model_dim[0], model_dim[1], w0=features_scales, is_first=True
        )
        other_layers = []
        for dim0, dim1 in zip(model_dim[1:-2], model_dim[2:-1]):
            other_layers.append(SirenLayer(dim0, dim1))
            # other_layers.append(nn.LayerNorm(dim1))
        final_layer = SirenLayer(model_dim[-2], model_dim[-1], is_last=True)
        self.model = nn.Sequential(first_layer, *other_layers, final_layer)

    def forward(self, xin):
        return self.model(xin)


class SIREN(RFF):
    def gen_architecture(self):
        self.mlp = SIREN_model(
            self.input_size,
            self.output_size,
            self.hp.model["hidden_nlayers"],
            self.hp.model["hidden_width"],
            self.hp.model["scale"],
        )

    def setup(self):
        self.first = nn.Identity()
