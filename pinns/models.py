from __future__ import annotations

import torch
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .parser import AttrDict

from .model_utils import linear_fn, RFFLayer, SkipLayer, MFN, ModifiedMLP, KAN


class INR(nn.Module):
    """Implicit Neural Representation model.

    A configurable neural network for learning continuous representations
    of signals (fields, images, PDE solutions). Supports multiple
    architectures: SIREN, RFF, WIRES, MFN, and KAN.

    Parameters
    ----------
    name : str
        Architecture name. One of 'SIREN', 'RFF', 'WIRES', 'MFN', 'KAN'.
    input_size : int
        Dimensionality of input coordinates (e.g., 2 for (t, x), 3 for (t, x, y)).
    output_size : int
        Dimensionality of output field (e.g., 1 for scalar fields).
    hp : AttrDict
        Hyperparameter dictionary. Must contain 'model' key with architecture
        configuration (hidden_nlayers, hidden_width, activation, etc.).

    Examples
    --------
    >>> hp = read_yaml('default-parameters.yml')
    >>> model = INR('SIREN', input_size=2, output_size=1, hp=hp)
    >>> x = torch.randn(100, 2)
    >>> y = model(x)  # shape: (100, 1)
    """

    def __init__(
        self,
        name: str,
        input_size: int,
        output_size: int,
        hp: AttrDict,
    ):
        super(INR, self).__init__()
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.hp = hp
        self.setup()
        self.gen_architecture()

    def setup(self):
        """Configure the activation function based on model architecture.

        Sets self.act to the appropriate nn.Module class for RFF and MFN
        architectures (Tanh or ReLU), or None for SIREN/WIRES/KAN.
        """
        if self.name == "RFF" or self.name == "MFN":
            if self.hp.model["activation"] == "tanh":
                self.act = nn.Tanh
            elif self.hp.model["activation"] == "relu":
                self.act = nn.ReLU
        else:
            self.act = None

    def gen_kan(self):
        """Build the KAN architecture.

        Constructs layer widths as [input_size, hidden_width, ..., output_size]
        and instantiates a KAN network.
        """
        width_std = self.hp.model["hidden_width"]
        layer_width = (
            [self.input_size]
            + [width_std for i in range(self.hp.model["hidden_nlayers"])]
            + [self.output_size]
        )
        # self.mlp = KAN(layer_width, grid=5, k=3, grid_eps=1.0, noise_scale_base=0.25)
        self.mlp = KAN(layer_width)

    def gen_architecture(self):
        """Dispatch to the appropriate architecture builder (KAN or MLP)."""
        if self.name == "KAN":
            self.gen_kan()
        else:
            self.gen_mlp_network()

    def gen_mlp_network(self):
        """Build the MLP-based architecture (SIREN, RFF, WIRES, MFN).

        Constructs layers using the factory from linear_fn(), optionally
        adding skip connections, ModifiedMLP wrapping, or MFN wrapping.
        """
        linear_layer_fn = linear_fn(self.hp.model["linear"], self.hp, self.act)
        layers = []
        width_std = self.hp.model["hidden_width"]
        layer_width = (
            [self.input_size]
            + [width_std for i in range(self.hp.model["hidden_nlayers"])]
            + [self.output_size]
        )
        last = len(layer_width)
        if self.name == "RFF":
            layer_width[1] = self.hp.model["mapping_size"]
        for i, width_i in enumerate(layer_width[:-1]):
            is_last = i == last - 2

            if i == 0 and self.name == "RFF":
                layer = RFFLayer(width_i, layer_width[i + 1], self.hp.model["scale"])
            elif i == 0 and self.name in ["SIREN", "WIRES", "MFN"]:
                layer = linear_layer_fn(width_i, layer_width[i + 1], is_first=True)
            else:
                layer = linear_layer_fn(width_i, layer_width[i + 1], is_last=is_last)
            if self.hp.model["skip"] and i != 0 and not is_last:
                layer = SkipLayer(layer)
            layers.append(layer)

        self.mlp = nn.Sequential(*layers)

        if self.hp.model["modified_mlp"]:
            self.mlp = ModifiedMLP(self.mlp, nn.Tanh, self.hp)
        if self.name == "MFN":
            self.mlp = MFN(self.mlp, self.hp)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Parameters
        ----------
        *args : torch.Tensor
            One or more tensors to be concatenated along dim=1
            before being passed through the network.

        Returns
        -------
        torch.Tensor
            Network output of shape (batch_size, output_size).
        """
        xin = torch.cat(args, dim=1)
        return self.mlp(xin)
