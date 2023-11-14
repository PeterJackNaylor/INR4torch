import torch
import numpy as np
import pinns
from generate_data import get_dataset
from example_dataloader import return_dataset
from example_pde_model import Advection
from example_model import return_model_advection as model
import matplotlib.pylab as plt
import matplotlib.cm as cm

torch.set_float32_matmul_precision("high")
model_hp = pinns.read_yaml("../default-parameters.yml")
c = model_hp.c
real_u, real_t, real_x = get_dataset(c=c, n_t=200, n_x=128)
gpu = torch.cuda.is_available()
device = "cuda" if gpu else "cpu"

model_hp.data_u = real_u.copy()
model_hp.data_t = real_t.copy()
model_hp.data_x = real_x.copy()
model_hp.gpu = gpu
model_hp.c = c
model_hp.verbose = True
model_hp.pth_name = "test.pth"
model_hp.npz_name = "test.npz"

Model_cl = model(model_hp.hard_periodicity)

NN, model_hp = pinns.train(model_hp, Advection, return_dataset, Model_cl, gpu=gpu)

xx = NN.test_set.x
tt = NN.test_set.t
predictions = NN.test_loop()
if gpu:
    predictions = predictions.cpu()
predictions = predictions.reshape((xx.shape[0], tt.shape[0])).numpy()

gt = real_u

score = np.linalg.norm(gt - predictions) / np.linalg.norm(gt)


print("##########################")
print("#   Relative L2 error:   #")
print("#                        #")
print(f"#        {score:.6f}        #")
print("#                        #")
print("##########################")


im = plt.imshow(
    predictions,
    extent=[tt.min(), tt.max(), xx.min(), xx.max()],
    aspect="auto",
    cmap="jet",
)
plt.colorbar(im)
plt.savefig("plots/predictions.png")
plt.close()

im = plt.imshow(
    gt, extent=[tt.min(), tt.max(), xx.min(), xx.max()], aspect="auto", cmap="jet"
)
plt.colorbar(im)
plt.savefig("plots/ground_truth.png")
plt.close()

im = plt.imshow(
    np.abs(gt - predictions),
    extent=[tt.min(), tt.max(), xx.min(), xx.max()],
    aspect="auto",
    cmap="jet",
)
plt.colorbar(im)
plt.savefig("plots/absolute_error.png")
plt.close()

plt.plot(predictions[:, 0])
plt.savefig("plots/predictions_time_0.png")
plt.close()

plt.plot(gt[:, 0])
plt.savefig("plots/ground_truth_time_0.png")
plt.close()

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
