from pinns.models import INR
import torch
import torch.nn as nn
from torch import cos, sin


class PeriodicityLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.p_t = nn.Parameter(torch.Tensor(1))
        torch.nn.init.constant_(self.p_t, torch.pi)

    def forward(self, x_t):
        x = x_t[:, 0]
        t = x_t[:, 1]
        w_t = 2 * torch.pi / self.p_t
        w_x = 1 * torch.pi
        return torch.column_stack(
            [cos(w_t * t), sin(w_t * t), cos(w_x * x), sin(w_x * x)]
        )


class INR_hard_periodicity(INR):
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
        self.hard_period = PeriodicityLayer()

        if hp.normalise_targets:
            self.final_act = torch.tanh
        else:
            self.final_act = nn.Identity()

    def forward(self, *args):
        xin = torch.cat(args, axis=1)
        xin = self.hard_period(xin)
        x = self.first(xin)

        if self.hp.model["modified_mlp"]:
            Ux = self.U(x)
            Vx = self.V(x)
        l_i = self.hp.model["hidden_nlayers"]
        for i, layer in enumerate(self.layer_iterator()):
            if i == 0 or i == l_i:
                y = layer(x)
            else:
                if i % 2 == 1 and self.hp.model["skip"]:
                    y = layer(x) + x
                else:
                    y = layer(x)

            if self.hp.model["modified_mlp"] and i != l_i:
                x = torch.mul(Ux, y) + torch.mul(Vx, 1 - y)
            else:
                x = y
            if i == self.hp.model["hidden_nlayers"]:
                out = x

        return self.final_act(out)


def return_model_advection(hard_periodicity):
    if hard_periodicity:
        return INR_hard_periodicity
    else:
        return INR
