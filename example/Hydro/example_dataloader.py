import numpy as np
import torch
import pinns
import pandas as pd
from datetime import datetime
import random

name_mapping = {
    0: "ARAGUAIANA",
    1: "BARRA DO GARCAS",
    2: "CARACARAI",
    3: "CONCEICAO DO ARAGUAIA",
    4: "OBIDOS - LINIGRAFO",
    5: "BOM JESUS DA LAPA"
}

def load_data(hp):
    P = load_csv("data/Samp_Precipitation.csv", hp, "P")
    E_e = load_csv("data/Samp_Evapotranspiration_ERA.csv", hp, "E_era")
    E_g = load_csv("data/Samp_Evapotranspiration_GLEAM.csv", hp, "E_gleam")
    Q = load_csv("data/Samp_Discharge.csv", hp, "Q")
    DS_G  = load_csv("data/Samp_TWSA_GSFC.csv", hp, "DS_GSFC")
    DS_J  = load_csv("data/Samp_TWSA_JPL.csv", hp, "DS_JPL")
    on = ["T"] if hp.bassin != 6 else ["T", "B"]
    data = P.merge(E_e, on=on).merge(E_g, on=on).merge(DS_G, on=on).merge(DS_J, on=on).merge(Q, on=on)
    if hp.bassin == 6:
        data = data[["T", "B", "E_era", "E_gleam", "DS_GSFC", "DS_JPL", "Q"]]

    # print("reducing data")
    # data = data.loc[200:223]
    # # data = data.loc[16:39]
    # data = data.reset_index(drop=True)
    return data

def set_time(time_array):
    # for months
    time_array = np.arange(time_array.shape[0])
    return time_array

def prep_bassins(df, name):
    cols = df.columns
    dfs = []
    for i, col in enumerate(cols[1:]):
        bassin_df = df[[cols[0]] + [col]]
        bassin_df["Bassin"] = i
        bassin_df.columns = ["T", name, "B"]
        dfs.append(bassin_df)
    df = pd.concat(dfs, axis=0)
    return df[["T", "B", name]]

def load_csv(path, hp, name):
    table = pd.read_csv(path)
    table["timestamp"] = set_time(table["timestamp"])
    if hp.bassin != 6:
        bassin = ["timestamp", name_mapping[hp.bassin]]
        table = table[bassin]
        table.columns = ["T", name]
    else:
        table = prep_bassins(table, name)
    return table

def split_train_test(timeline, p):
    timeline = list(timeline.copy())
    np.random.shuffle(timeline)
    n = int(len(timeline) * p)
    train, test = timeline[:n], timeline[n:]
    return train, test

def set_input_variables(hp):
    # by default: T, P, E, DS
    # if
    hp.input_variables = 4
    var_pos = {
        "P": 0,
        "E": [1, 2],
        "DS": [3, 4],
        "Q": 5
    }
    next_pos = 5
    if hp.bassin == 6:
        var_pos["B"] = next_pos
        hp.input_variables += 1
        next_pos += 1
    # if hp.add_dq:
    #     var_pos["dq/dt"] = next_pos
    #     hp.input_variables += 1
    #     next_pos += 1
    # if hp.add_old_q:
    #     var_pos["Q(t-1)"] = next_pos
    #     hp.input_variables += 1
        # next_pos += 1
    hp.var_pos = var_pos
    hp.var_names = var_pos = {
        "P": 0,
        "E_era": 1,
        "E_gleam": 2,
        "DS_GSFC": 3,
        "DS_JPL": 4,
        "Q": 5
    }
    return hp

def invert_input_variables(dic):
    out = {}
    for key, value in dic.items():
        if isinstance(value, list):
            for el in value:
                out[el] = key
        else:
            out[value] = key
    return out

def create_data(hp):
    hp = set_input_variables(hp)
    hp.input_variables_reverse = invert_input_variables(hp.var_pos)
    hp.input_variables_reverse_pos = invert_input_variables(hp.var_names)
    data = hp.data
    # if hp.fill_in_interpolation_one:
    #     data = data.interpolate(limit=1, limit_area="inside")
    # if hp.add_dq:
    #     data["dq/dt"] = data["Q"] - data["Q"].shift(+1)
    # if hp.add_old_q:
    #     data["Q(t-1)"] = data["Q"].shift(+1)
    hp.data = data
    return hp

# def filter_data(hp):
#     data = hp.data
#     data = data.dropna()
#     data = data.reset_index(drop=True)
#     return data

def return_dataset(hp, gpu=True):
    hp.data = load_data(hp)
    hp = create_data(hp)
    # hp.data = filter_data(hp)
    hp.train_idx, hp.test_idx = split_train_test(hp.data.index, hp.p)

    data_train = PESQ(hp, nv_samples=None, nv_targets=None, test=False, gpu=gpu)
    data_test = PESQ(
        hp,
        nv_samples=data_train.nv_samples,
        nv_targets=data_train.nv_targets,
        test=True,
        gpu=gpu,
    )
    return data_train, data_test

# def set_column_index(d):
#     out = []
#     pos = 0
#     for key, value in d.items():
#         if isinstance(value, list):
#             out.append(value[0])
#         else:
#             out.append(value)
#         if key == "E":
#             E_pos = pos
#         if key == "DS":
#             DS_pos = pos
#         pos +=1
        
#     return out, E_pos, DS_pos

class PESQ(pinns.DataPlaceholder):
    def __init__(self, hp, nv_samples=None, nv_targets=None, test=True, gpu=True):

        self.hp = hp
        self.need_target = True
        self.output_size = 6
        idx = hp.train_idx if not test else hp.test_idx
        # T, P, E_era, E_gleam, DS_GSFC, DS_JPL, Q
        if hp.bassin == 6:
            samples =  hp.data[["T", "B"]].values[idx]
        else:
            samples =  hp.data[["T"]].values[idx]

        not_target = ["B", "T"]
        targets = hp.data[[el for el in hp.data.columns if el not in not_target]].values[idx]
        
        self.input_size = samples.shape[1] #because of the double E and double DS
        self.test = test
        self.bs = hp.losses["mse"]["bs"]
        normalise_targets = hp.normalise_targets

        # self.column_index, self.E_pos, self.DS_pos = set_column_index(hp.var_pos)
        samples = samples.astype(float)
        nv_samples = self.normalize(samples, nv_samples, True)
        if self.need_target:
            if not normalise_targets:
                nv_targets = [(0, 1) for _ in range(targets.shape[1])]
            nv_targets = self.normalize(targets, nv_targets, True)
        self.samples = torch.from_numpy(samples).float()
        self.nv_samples = nv_samples
        self.nv_targets = nv_targets
        if self.need_target:
            self.targets = torch.from_numpy(targets)

        self.setup_cuda(gpu)
        self.setup_batch_idx()

    def normalize(self, vector, nv, include_last=True):
        c = vector.shape[1]
        if nv is None:
            nv = []
            for i, vect in enumerate(vector.T):
                if i == c - 1 and not include_last:
                    break
                m = (np.nanmax(vect)+ np.nanmin(vect)) / 2
                s = (np.nanmax(vect) - np.nanmin(vect)) / 2
                nv.append((m, s))

        for i in range(c):
            if i == c - 1 and not include_last:
                break
            vector[:, i] = (vector[:, i] - nv[i][0]) / nv[i][1]

        return nv

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # randomly pick a E and DS
        if not self.need_target:
            # return sample
            return {"x": sample}
        target = self.targets[idx]
        return {"x": sample, "z": target}
