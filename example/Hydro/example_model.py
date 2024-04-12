from pinns.models import INR
import torch
import torch.nn as nn
from torch import cos, sin


class INR_Hydro(INR):
    
def return_model_advection(hard_periodicity):
    if hard_periodicity:
        return INR_hard_periodicity
    else:
        return INR
