import torch
import numpy as np
import pinns
from generate_data import get_dataset
from example_dataloader import return_dataset
from example_pde_model import Advection
import matplotlib.pylab as plt

c = 10
real_u, real_t, real_x = get_dataset(c=c, n_t=200, n_x=128)
gpu = torch.cuda.is_available()
device = "cuda" if gpu else "cpu"

model_hp = pinns.read_yaml("../default-parameters.yml")
model_hp.data_u = real_u.copy()
model_hp.data_t = real_t.copy()
model_hp.data_x = real_x.copy()
model_hp.gpu = gpu
model_hp.c = c
model_hp.verbose = True
model_hp.pth_name = "test.pth"
model_hp.npz_name = "test.npz"

NN, model_hp = pinns.train(model_hp, Advection, return_dataset, gpu=gpu)

xx = NN.test_set.x
tt = NN.test_set.t
predictions = NN.test_loop()
predictions = predictions.reshape((tt.shape[0], xx.shape[0])).T
gt = NN.test_set.targets.reshape((tt.shape[0], xx.shape[0])).T

if gpu:
    predictions = predictions.cpu()
    gt = gt.cpu()
plt.imshow(predictions, extent=[tt.min(), tt.max(), xx.min(), xx.max()], aspect="auto")
plt.savefig("plots/predictions.png")
plt.close()

plt.imshow(gt, extent=[tt.min(), tt.max(), xx.min(), xx.max()], aspect="auto")
plt.savefig("plots/ground_truth.png")
plt.close()

plt.plot(predictions[:, 0])
plt.savefig("plots/predictions_time_0.png")
plt.close()

plt.plot(gt[:, 0])
plt.savefig("plots/ground_truth_time_0.png")
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
    for k in NN.lambdas_scalar.keys():
        plt.plot(NN.lambdas_scalar[k], label=k)
    plt.legend()
    plt.savefig("plots/lambdas_scalar.png")
    plt.close()
except:
    print("Couldn't plot lambdas_scalar")

for key in NN.temporal_weights.keys():
    try:
        t_weights = torch.column_stack(NN.temporal_weights[key])
        if gpu:
            t_weights = t_weights.cpu()
        for k in range(t_weights.shape[0]):
            plt.plot(t_weights[k], label=f"w_{k}")
        plt.legend()
        plt.savefig(f"plots/w_temp_{key}_weights.png")
        plt.close()
    except:
        print(f"Couldn't plot t_weights for {key}")