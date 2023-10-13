import numpy as np

# from .data_XYTZ import return_dataset
from .models import ReturnModel, gen_b

# from .util_train import estimate_density

# import torch


def train(
    hp, estimate_density_cl, dataset_fn, trial=None, return_model=True, gpu=False
):

    train, test = dataset_fn(hp, gpu=gpu)

    hp.input_size = train.input_size
    hp.output_size = len(train.nv_targets)
    hp.nv_samples = train.nv_samples
    hp.nv_targets = train.nv_targets

    if hp.model["name"] == "RFF":
        hp.B = gen_b(
            hp.model["mapping_size"], hp.model["scale"], hp.input_size, gpu=gpu
        )

    model = ReturnModel(
        hp.input_size,
        output_size=hp.output_size,
        hp=hp,
    )

    if gpu:
        model = model.cuda()

    NN = estimate_density_cl(train, test, model, hp, gpu, trial=trial)
    NN.fit()

    if "B" in hp.keys():
        hp.B = np.array(hp.B.cpu())
    if hp.save_model:
        np.savez(
            hp.npz_name,
            **hp,
        )

    return NN, hp
