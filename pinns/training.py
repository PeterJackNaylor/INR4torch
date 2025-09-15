import numpy as np
import torch

def check_model_hp(hp):
    if "linear" not in hp.model:
        hp.model["linear"] = "HE"
    if "eps" not in hp:
        hp.eps = 1.e-8
    if "clip_gradients" not in hp:
        hp.clip_gradients = True
    if "cosine_anealing" not in hp:
        hp.cosine_anealing = {"status": False, "min_eta": 0, "step": 500}
    return hp

def check_data_loader_hp(hp):
    if "model" in hp:
        if "name" not in hp.model:
            hp.model["name"] = "default"
    else:
        hp.model = {"name": "default"}
    if "hard_periodicity" not in hp:
        hp.hard_periodicity = False
    return hp

def check_estimator_hp(hp):
    if "verbose" not in hp:
        hp.verbose = True
    if "optuna" not in hp:
        hp.optuna = {"patience": 10000, "trials": 1}
    if "save_model" not in hp:
        hp.save_model = False
    # if "npz_name" not in hp:
    #     hp.npz_name = "default.npz"
    # if "pth_name" not in hp:
    #     hp.pth_name = "default.pth"
    return hp   

def train(
    hp,
    estimate_density_cl,
    dataset_fn,
    model_cl,
    initial_weights=None,
    trial=None,
    gpu=False,
):
    hp = check_data_loader_hp(hp)
    hp = check_model_hp(hp)
    hp = check_estimator_hp(hp)

    train, test = dataset_fn(hp, gpu=gpu)

    hp.input_size = train.input_size
    hp.output_size = train.output_size
    hp.nv_samples = train.nv_samples
    hp.nv_targets = train.nv_targets

    model = model_cl(
        hp.model["name"],
        hp.input_size,
        output_size=hp.output_size,
        hp=hp,
    )
    # model = torch.compile(model, backend="inductor")
    if gpu:
        model = model.cuda()

    NN = estimate_density_cl(train, test, model, hp, gpu, trial=trial)
    if initial_weights:
        NN.model.load_state_dict(torch.load(initial_weights, map_location=NN.device))
    NN.fit()

    if hp.save_model:
        np.savez(
            hp.npz_name,
            **hp,
        )

    return NN, hp
