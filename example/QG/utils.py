import os
import torch
from tqdm import tqdm
import matplotlib.pylab as plt
import numpy as np
import matplotlib.cm as cm


def CheckOrCreate(folder):
    try:
        os.mkdir(folder)
    except:
        pass

def predict(array, model):
    n_data = array.shape[0]
    verbose = model.hp.verbose
    bs = model.hp.losses["mse"]["bs"]
    batch_idx = torch.arange(0, n_data, dtype=int, device=model.device)
    range_ = range(0, n_data, bs)
    train_iterator = tqdm(range_) if verbose else range_
    preds = []
    with torch.no_grad():
        with torch.autocast(
            device_type=model.device, dtype=model.dtype, enabled=model.use_amp
        ):
            for i in train_iterator:
                idx = batch_idx[i : (i + bs)]
                samples = array[idx]
                pred = model.model(samples)
                preds.append(pred)
            if i + bs < n_data:
                idx = batch_idx[(i + bs) :]
                samples = array[idx]
                pred = model.model(samples)
                preds.append(pred)
    preds = torch.cat(preds)
    return preds

def model_plot(NN, samples, targets, model_hp, ts, x, y, z, L, interval):
    n = len(NN.test_scores)
    f = model_hp.test_frequency
    plt.plot(list(range(1 * f, (n + 1) * f, f)), NN.test_scores)
    plt.savefig("single_run/train_health/test_scores.png")
    plt.close()

    for k in NN.loss_values.keys():
        try:
            loss_k = NN.loss_values[k]
            plt.plot(loss_k)
            plt.savefig(f"single_run/train_health/{k}.png")
            plt.close()
        except:
            print(f"Couldn't plot {k}")
    try:
        plt.plot([np.log(lr) / np.log(10) for lr in NN.lr_list])
        plt.savefig("single_run/train_health/LR.png")
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
        plt.savefig("single_run/train_health/lambdas_scalar.png")
        plt.close()
    except:
        print("Couldn't plot lambdas_scalar")

    for key in NN.temporal_weights.keys():
        try:
            f = model_hp.temporal_causality["step"]
            t_weights = torch.column_stack(NN.temporal_weights[key])
            x_axis = t_weights.shape[1]  # because we will remove the first one
            x_axis = list(range(0, x_axis * f, f))
            import pdb; pdb.set_trace()
            if "cuda" in str(t_weights.device):
                t_weights = t_weights.cpu()
            color = cm.hsv(np.linspace(0, 1, t_weights.shape[0]))
            for k in range(t_weights.shape[0]):
                plt.plot(x_axis, t_weights[k], label=f"w_{k}", color=color[k])
            plt.legend()
            plt.savefig(f"single_run/train_health/w_temp_{key}_weights.png")
            plt.close()
        except:
            print(f"Couldn't plot t_weights for {key}")

    vlim = (-np.abs(targets).max(), np.abs(targets).max())
    for t in list(ts):
        x = samples[:, 0]
        y = samples[:, 1]
        extent = [x.min(), x.max(), y.min(), y.max()]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        t_idx = int(round(t / interval))
        ax1.title.set_text("GT points")
        ax2.title.set_text("GT grids")
        ax3.title.set_text("GT overlay")
        ax1.scatter(x, y, c=targets, s=3.5, vmin=vlim[0], vmax=vlim[1], cmap="RdBu_r")
        ax1.set_xlim([0, L])
        ax1.set_ylim([0, L])
        ax2.imshow(z[::-1, :, t_idx], vmin=vlim[0], vmax=vlim[1], extent=extent, cmap="RdBu_r")
        ax3.scatter(x, y, c=targets, s=3.5, vmin=vlim[0], vmax=vlim[1], cmap="RdBu_r")
        ax3.imshow(z[::-1, :, t_idx], vmin=vlim[0], vmax=vlim[1], extent=extent, cmap="RdBu_r")
        fig.savefig(f"single_run/train_health/checking_scatter_time_{round(t,1)}.png")
        plt.close()

