import torch
import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle

import pinns

from example_dataloader import return_dataset
from example_pde_model import Hydro, conservation_residue
from pinns.models import INR

torch.set_float32_matmul_precision("high")
model_hp = pinns.read_yaml("hydro.yml")
gpu = torch.cuda.is_available()
device = "cuda" if gpu else "cpu"

model_hp.gpu = gpu
model_hp.verbose = True
model_hp.pth_name = "test.pth"
model_hp.npz_name = "test.npz"

NN, model_hp = pinns.train(model_hp, Hydro, return_dataset, INR, gpu=gpu)
time_input = np.arange(model_hp.data["T"].min(), model_hp.data["T"].max()+1)

tt = (NN.test_set.samples[:,0].cpu().numpy() * NN.data.nv_samples[0][1] + NN.data.nv_samples[0][0]).astype(int) - model_hp.data["T"].min()
is_test = np.zeros_like(time_input, dtype=bool)

is_test[tt] = True

gt = model_hp["data"].values[:, 1:]
NN.test_set.samples = torch.tensor((time_input[...,np.newaxis] - NN.data.nv_samples[0][0]) / NN.data.nv_samples[0][1])

dtype = torch.float16 if model_hp.gpu else torch.bfloat16
device = "cuda" if model_hp.gpu else "cpu"
NN.test_set.samples = NN.test_set.samples.to(device, dtype=dtype)
NN.n_test = len(NN.test_set)
with torch.no_grad():
    predictions = NN.test_loop()
if model_hp.gpu:
    predictions = predictions.cpu()
predictions = predictions.numpy()

inverse_dic = model_hp.input_variables_reverse_pos

for i in range(NN.n_outputs):
    fig, ax = plt.subplots()
    fig.set_figwidth(20)
    predictions[:, i] = predictions[:, i] * NN.data.nv_targets[i][1] + NN.data.nv_targets[i][0]
    ax.plot(time_input, predictions[:, i], linewidth=2.0, label=f"{inverse_dic[i]} hat", marker='o')
    ax.plot(time_input, gt[:, i], linewidth=2.0, label=f"{inverse_dic[i]}", marker='x')
    mini = min(predictions[:, i].min(), np.nanmin(gt[:, i]))
    maxi = max(predictions[:, i].max(), np.nanmax(gt[:, i]))
    for k in list(tt):
        k += model_hp.data["T"].min()
        ax.add_patch(Rectangle((k-0.5,mini), 1., maxi-mini,
                    edgecolor='none',
                    facecolor='red',
                    alpha=0.2,
                    lw=1))
    ax.legend()
    plt.savefig(f"estimation/{inverse_dic[i]}.png")
    plt.close()

residues = conservation_residue(NN.model, NN.test_set.samples.to(torch.float32), model_hp.nv_samples, model_hp.nv_targets, model_hp.var_pos)
if model_hp.gpu:
    residues = residues.cpu()
residues = residues.detach().numpy()
fig, ax = plt.subplots()
fig.set_figwidth(20)
ax.plot(time_input, residues, linewidth=2.0, label=f"residue hat", marker='o')
mini, maxi = min(residues), max(residues)
for k in list(tt):
    k += model_hp.data["T"].min()
    ax.add_patch(Rectangle((k-0.5,mini), 1., maxi-mini,
                edgecolor='none',
                facecolor='red',
                alpha=0.2,
                lw=1))
ax.legend()
plt.savefig(f"estimation/residues.png")
plt.close()

# gt = real_u
def nannorm(x):
    return np.nansum(x ** 2) ** 0.5 
score = nannorm(gt - predictions) / nannorm(gt)


print("##########################")
print("# Relative L2 error (all)#")
print("#                        #")
print(f"#        {score:.6f}        #")
print("#                        #")
print("##########################")



n = len(NN.test_scores)
f = model_hp.test_frequency
plt.plot(list(range(1 * f, (n + 1) * f, f)), NN.test_scores)
plt.savefig("plots/test_scores.png")
plt.close()

for k in NN.loss_values.keys():
    try:
        loss_k = NN.loss_values[k]
        plt.plot(loss_k)
        plt.savefig(f"plots/{k}.png")
        plt.close()
    except:
        print(f"Couldn't plot {k}")
try:
    plt.plot([np.log(lr) / np.log(10) for lr in NN.lr_list])
    plt.savefig("plots/LR.png")
    plt.close()
except:
    print("Coulnd't plot LR")
try:
    if model_hp.relobralo["status"]:
        f = model_hp.relobralo["step"]
    elif model_hp.self_adapting_loss_balancing["status"]:
        f = model_hp.self_adapting_loss_balancing["step"]

    for k in NN.lambdas_scalar.keys():
        n = len(NN.lambdas_scalar[k])
        plt.plot(list(range(0, n * f, f)), NN.lambdas_scalar[k], label=k)
    plt.legend()
    plt.savefig("plots/lambdas_scalar.png")
    plt.close()
except:
    print("Couldn't plot lambdas_scalar")

for key in NN.temporal_weights.keys():
    try:
        f = model_hp.temporal_causality["step"]
        t_weights = torch.column_stack(NN.temporal_weights[key])
        x_axis = t_weights.shape[1]  # because we will remove the first one
        x_axis = list(range(0, x_axis * f, f))
        if gpu:
            t_weights = t_weights.cpu()
        color = cm.hsv(np.linspace(0, 1, t_weights.shape[0]))
        for k in range(t_weights.shape[0]):
            plt.plot(x_axis, t_weights[k], label=f"w_{k}", color=color[k])
        plt.legend()
        plt.savefig(f"plots/w_temp_{key}_weights.png")
        plt.close()
    except:
        print(f"Couldn't plot t_weights for {key}")
