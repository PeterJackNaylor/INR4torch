from pinns.models import INR
import torch
import torch.nn as nn
from torch import cos, sin
from functools import partial

def apply_combination_w(tensor, positions, weights, softmax, mean, std):
    output = tensor[:, positions]
    output = output * mean + std
    lin_comb = softmax(weights)
    return (output * lin_comb).sum(axis=1)

def apply_combination(tensor, positions, mean, std):
    return tensor[:, positions] * std + mean

def get_nv_matrix(nv):
    mean = torch.ones(len(nv))
    std = torch.ones(len(nv))
    for i in range(len(nv)):
        mean[i] = nv[i][0]
        std[i] = nv[i][1]
    return mean, std

class INR_Hydro_forecast_combi(INR):
    def __init__(
        self,
        name,
        input_size,
        output_size,
        hp,
    ):
        super(INR_Hydro_forecast_combi, self).__init__(
            name,
            input_size,
            output_size,
            hp
        )
        
        for var in self.hp.var_pos:
            if isinstance(self.hp.var_pos[var], list):
                size = len(self.hp.var_pos[var])
                self.__setattr__(f"{var}_weights", nn.Parameter(torch.rand(1, size)))
                self.__setattr__(f"{var}_softmax", torch.nn.Softmax(dim=1))
                mean, std = get_nv_matrix(self.hp.nv_targets[self.hp.var_pos[var][0]:self.hp.var_pos[var][-1]+1])
                if self.hp.gpu:
                    mean, std = mean.to("cuda"), std.to("cuda")
                var_func = partial(apply_combination_w, positions=self.hp.var_pos[var], weights=getattr(self, f"{var}_weights"), softmax=getattr(self, f"{var}_softmax"), mean=mean, std=std)
                self.__setattr__(f"{var}_func", var_func)
            else:
                mean, std = get_nv_matrix([self.hp.nv_targets[self.hp.var_pos[var]]])
                if self.hp.gpu:
                    mean, std = mean.to("cuda"), std.to("cuda")
                var_func = partial(apply_combination, positions=self.hp.var_pos[var], mean=mean, std=std)
                self.__setattr__(f"{var}_func", var_func)
                
    def forward(self, *args):
        xin = torch.cat(args, axis=1)
        hat_inputs = self.mlp(xin)
        out = []
        for var in self.hp.var_pos:
            out.append(getattr(self, f"{var}_func")(hat_inputs))
        out = torch.stack(out).T
        return hat_inputs, out


class LSTM_INR_Hydro_forecast_combi(INR_Hydro_forecast_combi):
    def __init__(
        self,
        name,
        input_size,
        output_size,
        hp,
    ):
        super(LSTM_INR_Hydro_forecast_combi, self).__init__(
            name,
            input_size,
            output_size,
            hp
        )
        self.LSTM = False
        
    def forward(self, *args):
        xin = torch.cat(args, axis=1)
        hat_inputs = self.mlp(xin)
        out = []
        for var in self.hp.var_pos:
            out.append(getattr(self, f"{var}_func")(hat_inputs))
        out = torch.stack(out).T
        return hat_inputs, out

