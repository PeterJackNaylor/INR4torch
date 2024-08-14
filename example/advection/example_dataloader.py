import numpy as np
import torch
import pinns


def return_dataset(hp, gpu=True):
    data_train = XT(hp, nv_samples=None, nv_targets=None, test=False, gpu=gpu)
    data_test = XT(
        hp,
        nv_samples=data_train.nv_samples,
        nv_targets=data_train.nv_targets,
        test=True,
        gpu=gpu,
    )
    if hp.hard_periodicity:
        data_train.input_size = 2 * data_train.input_size
        data_test.input_size = 2 * data_test.input_size
    return data_train, data_test


class XT(pinns.DataPlaceholder):
    def __init__(self, hp, nv_samples=None, nv_targets=None, test=True, gpu=True):
        self.name = hp.model["name"]
        self.need_target = True
        self.input_size = 2
        self.output_size = 1
        self.test = test
        self.bs = hp.losses["mse"]["bs"]
        normalise_targets = hp.normalise_targets
        u = np.float32(hp.data_u)
        t = np.float32(hp.data_t)
        x = np.float32(hp.data_x)
        self.t = t
        self.x = x
        if test:
            tt, xx = np.meshgrid(t, x)
            xx = np.flip(xx, axis=0)
            samples = np.vstack([xx.ravel(), tt.ravel()]).T
            targets = u.ravel().reshape(samples.shape[0], 1)
        else:
            g_x = np.sin(x)
            t_0 = np.zeros_like(x)
            samples = np.vstack([x, t_0]).T
            targets = g_x.reshape(samples.shape[0], 1)

        nv_samples = self.normalize(samples, nv_samples, False)
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

    def setup_cuda(self, gpu):
        if gpu:
            dtype = torch.float16
            if self.name == "WIRES":
                dtype = torch.float32
            device = "cuda"
        else:
            dtype = torch.bfloat16
            if self.name == "WIRES":
                dtype = torch.bfloat32
            device = "cpu"
        if self.name == "WIRES":
            dtype = dtype = torch.float32

        self.samples = self.samples.to(device, dtype=dtype)
        if self.need_target:
            self.targets = self.targets.to(device, dtype=dtype)
        self.device = device
        self.dtype = dtype
