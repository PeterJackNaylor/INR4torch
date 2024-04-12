import torch
import torch.nn as nn
from .model_utils import linear_fn, RFFLayer, SkipLayer, MFN, ModifiedMLP

class INR(nn.Module):
    def __init__(
        self,
        name,
        input_size,
        output_size,
        hp,
    ):
        super(INR, self).__init__()
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.hp = hp
        self.setup()

        self.gen_architecture()

    def setup(self):
        if self.name == "RFF" or self.name == "MFN":
            if self.hp.model["activation"] == "tanh":
                self.act = nn.Tanh
            elif self.hp.model["activation"] == "relu":
                self.act = nn.ReLU
        else:
            self.act = None

    def gen_architecture(self):
        linear_layer_fn = linear_fn(self.hp.model["linear"], self.hp, self.act)
        layers = []
        width_std = self.hp.model["hidden_width"]
        layer_width = [self.input_size] + [
            width_std for i in range(self.hp.model["hidden_nlayers"])
        ] + [self.output_size]
        last = len(layer_width)
        if self.name == "RFF":
            layer_width[1] = self.hp.model["mapping_size"]
        for i, width_i in enumerate(layer_width[:-1]):
            is_last = i == last - 2
            
            if i == 0 and self.name == "RFF":
                layer = RFFLayer(width_i, layer_width[i+1], self.hp.model["scale"])
            elif i == 0 and self.name in ["SIREN", "WIRES", "MFN"]:
                layer = linear_layer_fn(width_i, layer_width[i+1], is_first=True)
            else:
                layer = linear_layer_fn(width_i, layer_width[i+1], is_last=is_last)
            if self.hp.model["skip"] and i != 0 and not is_last:
                layer = SkipLayer(layer)
            layers.append(layer)

        self.mlp = nn.Sequential(*layers)

        if self.hp.model["modified_mlp"]:
            self.mlp = ModifiedMLP(self.mlp, nn.Tanh, self.hp)
        if self.name == "MFN":
            self.mlp = MFN(self.mlp, self.hp)

    def forward(self, *args):
        xin = torch.cat(args, axis=1)
        return self.mlp(xin)
