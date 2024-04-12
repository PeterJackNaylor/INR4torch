import numpy as np
import torch

# from .models import gen_b


def train(
    hp,
    estimate_density_cl,
    dataset_fn,
    model_cl,
    initial_weights=None,
    trial=None,
    gpu=False,
):
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
