import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pylab as plt
from math import ceil
import pinns
from example_dataloader import return_dataset
from example_pde_model import QGSurface
from pinns.models import INR
from utils import predict, model_plot, CheckOrCreate


def load_data(path):
    names = ["x_values", "y_values", "psi_values", "t_values", "qg_data"]
    # return [np.load(f"{path}/{n}.npy") for n in names]
    return [np.load(f"pyqg_out/{n}.npy") for n in names]


torch.set_float32_matmul_precision("high")
model_hp = pinns.read_yaml("QG.yml")

x, y, z, time, xytz = load_data("./data")
bmax = 4.05
bmin = 1.05  # 3.45
xytz = xytz[(xytz[:, 2] < bmax) & (xytz[:, 2] > bmin)]

z = z[:, :, (time < bmax) & (time > bmin)]
time = time[(time < bmax) & (time > bmin)]
gpu = torch.cuda.is_available()
device = "cuda" if gpu else "cpu"

n = xytz.shape[0]
bs = model_hp.losses["mse"]["bs"]
model_hp.max_iters = ceil(n // bs) * model_hp.epochs
model_hp.test_frequency = ceil(n // bs) * model_hp.test_epochs
model_hp.learning_rate_decay["step"] = (
    ceil(n // bs) * model_hp.learning_rate_decay["epoch"]
)
model_hp.cosine_anealing["step"] = ceil(n // bs) * model_hp.cosine_anealing["epoch"]
model_hp.self_adapting_loss_balancing["step"] = (
    ceil(n // bs) * model_hp.self_adapting_loss_balancing["epoch"]
)
model_hp.data_x = x.copy()
model_hp.data_y = y.copy()
model_hp.data_t = time.copy()
model_hp.data_z = z.copy()
model_hp.data_xytz = xytz.copy()

model_hp.gpu = gpu
model_hp.verbose = True
model_hp.pth_name = "test.pth"
model_hp.npz_name = "test.npz"


def mse(y, yh):
    return np.sqrt(((y - yh) ** 2).mean())


NN, model_hp = pinns.train(model_hp, QGSurface, return_dataset, INR, gpu=gpu)

xx = NN.test_set.x
yy = NN.test_set.y
L = yy.max()
predictions = NN.test_loop()
if gpu:
    predictions = predictions.cpu()
predictions = (
    predictions.numpy().astype(float) * NN.test_set.nv_targets[0][1]
    + NN.test_set.nv_targets[0][0]
)
pred_cube = predictions.reshape(len(NN.test_set.time_idx), 256, 256)
pred_cube = np.transpose(pred_cube, (1, 2, 0))
print(
    "RMSE   ",
    mse(z[:, :, NN.test_set.time_idx], pred_cube) / NN.test_set.nv_targets[0][1],
)

score = np.linalg.norm(z[:, :, NN.test_set.time_idx] - pred_cube) / np.linalg.norm(
    z[:, :, NN.test_set.time_idx]
)


print("##########################")
print("#   Relative L2 error:   #")
print("#                        #")
print(f"#        {score:.6f}        #")
print("#                        #")
print("##########################")


samples = NN.data.samples.cpu().numpy().copy()
targets = (
    NN.data.targets.cpu().numpy().copy() * NN.test_set.nv_targets[0][1]
    + NN.test_set.nv_targets[0][0]
)
for i in range(3):
    samples[:, i] = (
        samples[:, i] * NN.test_set.nv_samples[i][1] + NN.test_set.nv_samples[i][0]
    )
ts = np.unique(xytz[:, 2])
interval = 0.1 

train_predictions = (
    predict(NN.data.samples[:, :3], NN).cpu().numpy() * NN.test_set.nv_targets[0][1]
    + NN.test_set.nv_targets[0][0]
)
train_targets = (
    NN.data.targets.cpu().numpy() * NN.test_set.nv_targets[0][1]
    + NN.test_set.nv_targets[0][0]
)


test_targets = (
    NN.test_set.targets.cpu().numpy() * NN.test_set.nv_targets[0][1]
    + NN.test_set.nv_targets[0][0]
)
print(
    "Training RMSE:",
    mse(train_predictions, train_targets) / NN.test_set.nv_targets[0][1],
)

CheckOrCreate("single_run")
CheckOrCreate("single_run/training")
pointsize = 2
np.abs(targets).max()
vlim = (-np.abs(targets).max(), np.abs(targets).max())
for i, t in enumerate(NN.test_set.time_idx):
    s = x.shape[0] * y.shape[0]
    actual_time = time[t]
    idx = (samples[:, 2] > actual_time - interval / 2) & (
        samples[:, 2] < actual_time + interval / 2
    )
    pred = train_predictions[idx]
    target = targets[idx]
    xaxis = samples[idx, 0]
    yaxis = samples[idx, 1]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    ax1.title.set_text("GT")
    ax2.title.set_text("Pred")
    ax3.title.set_text("Diff GT-PRED")
    im1 = ax1.scatter(
        xaxis, yaxis, c=target, cmap="RdBu_r", s=pointsize, vmin=vlim[0], vmax=vlim[1]
    )
    ax1.set_xlim([0, L])
    ax1.set_ylim([0, L])
    plt.colorbar(im1)

    im2 = ax2.scatter(
        xaxis, yaxis, c=pred, cmap="RdBu_r", s=pointsize, vmin=vlim[0], vmax=vlim[1]
    )
    ax2.set_xlim([0, L])
    ax2.set_ylim([0, L])

    plt.colorbar(im2)

    im3 = ax3.scatter(
        xaxis,
        yaxis,
        c=np.abs(target - pred),
        cmap="RdBu_r",
        s=pointsize,
        vmin=0,
        vmax=np.abs(target - pred).max(),
    )
    ax3.set_xlim([0, L])
    ax3.set_ylim([0, L])

    plt.colorbar(im3)

    fig.savefig(f"single_run/training/train_prediction_time_{actual_time}.png")

    plt.close()

CheckOrCreate("single_run")
CheckOrCreate("single_run/prediction")
CheckOrCreate("single_run/train_health")
for i, t in enumerate(NN.test_set.time_idx):
    s = x.shape[0] * y.shape[0]
    pred_t = pred_cube[::-1, :, i]
    actual_time = time[t]
    gt_t = z[::-1, :, t]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    ax1.title.set_text("GT")
    ax2.title.set_text("Pred")
    ax3.title.set_text("Diff GT-PRED")
    extent = [x.min(), x.max(), y.min(), y.max()]
    im1 = ax1.imshow(
        gt_t, extent=extent, aspect="auto", cmap="RdBu_r", vmin=vlim[0], vmax=vlim[1]
    )
    plt.colorbar(im1)

    im2 = ax2.imshow(
        pred_t, extent=extent, aspect="auto", cmap="RdBu_r", vmin=vlim[0], vmax=vlim[1]
    )
    plt.colorbar(im2)

    im3 = plt.imshow(
        np.abs(gt_t - pred_t),
        extent=extent,
        aspect="auto",
        cmap="RdBu_r",
        vmin=0,
        vmax=np.abs(gt_t - pred_t).max(),
    )
    plt.colorbar(im3)
    fig.savefig(f"single_run/prediction/prediction_time_{actual_time}.png")

    plt.close()


# plt.plot(predictions[:, 0])
# plt.savefig("plots/predictions_time_0.png")
# plt.close()

# plt.plot(gt[:, 0])
# plt.savefig("plots/ground_truth_time_0.png")
# plt.close()
model_plot(NN, samples, targets, model_hp, ts, x , y, z, L, interval)