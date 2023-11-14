from functools import partial
import torch
import numpy as np
import pinns
from generate_data import get_dataset
from example_dataloader import return_dataset
from example_pde_model import Advection
from example_model import return_model_advection as model_fn
import matplotlib.pylab as plt
import optuna


def objective(trial, model_hp, real_u):
    model_hp.pth_name = f"multiple/test_{trial.number}.pth"
    model_hp.npz_name = f"multiple/test_{trial.number}.npz"
    Model = model_fn(model_hp.hard_periodicity)

    NN, model_hp = pinns.train(
        model_hp, Advection, return_dataset, Model, gpu=gpu, trial=trial
    )
    xx = NN.test_set.x
    tt = NN.test_set.t
    predictions = NN.test_loop()
    gt = real_u
    if gpu:
        predictions = predictions.cpu()
    predictions = predictions.reshape((xx.shape[0], tt.shape[0])).numpy()
    score = np.linalg.norm(gt - predictions) / np.linalg.norm(gt)
    return score


def load_model(model_hp, weights):
    Model = model_fn(model_hp.hard_periodicity)
    model_hp.B = torch.from_numpy(model_hp.B)
    model = Model(
        model_hp.model["name"],
        model_hp.input_size,
        output_size=model_hp.output_size,
        hp=model_hp,
    )
    if gpu:
        model = model.cuda()

    model.load_state_dict(torch.load(weights, map_location=device))

    train, test = return_dataset(model_hp, gpu=gpu)

    model_hp.input_size = train.input_size
    model_hp.output_size = len(train.nv_targets)
    model_hp.nv_samples = train.nv_samples
    model_hp.nv_targets = train.nv_targets

    NN = Advection(train, test, model, model_hp, gpu)
    return NN


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

objective = partial(objective, model_hp=model_hp, real_u=real_u)

study = optuna.create_study()
study.optimize(objective, n_trials=model_hp.optuna["trials"])


id_trial = study.best_trial.number

weights = f"multiple/test_{id_trial}.pth"

NN = load_model(model_hp, weights)

xx = NN.test_set.x
tt = NN.test_set.t
predictions = NN.test_loop()

gt = real_u
if gpu:
    predictions = predictions.cpu()
predictions = predictions.reshape((xx.shape[0], tt.shape[0])).numpy()
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
plt.savefig("multiple/plots/predictions.png")
plt.close()

im = plt.imshow(
    gt, extent=[tt.min(), tt.max(), xx.min(), xx.max()], aspect="auto", cmap="jet"
)
plt.colorbar(im)
plt.savefig("multiple/plots/ground_truth.png")
plt.close()

im = plt.imshow(
    np.abs(gt - predictions),
    extent=[tt.min(), tt.max(), xx.min(), xx.max()],
    aspect="auto",
    cmap="jet",
)
plt.colorbar(im)
plt.savefig("multiple/plots/absolute_error.png")
plt.close()

plt.plot(predictions[:, 0])
plt.savefig("multiple/plots/predictions_time_0.png")
plt.close()

plt.plot(gt[:, 0])
plt.savefig("multiple/plots/ground_truth_time_0.png")
plt.close()
