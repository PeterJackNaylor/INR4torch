import numpy as np
import torch
import pinns


def return_dataset(hp, gpu=True):
    data_train = XYT(hp, nv_samples=None, nv_targets=None, test=False, gpu=gpu)
    data_test = XYT(
        hp,
        nv_samples=data_train.nv_samples,
        nv_targets=data_train.nv_targets,
        test=True,
        gpu=gpu,
    )
    return data_train, data_test


class XYT(pinns.DataPlaceholder):
    def __init__(self, hp, nv_samples=None, nv_targets=None, test=True, gpu=True):
        self.need_target = True
        self.input_size = 3
        self.test = test
        self.bs = hp.losses["mse"]["bs"]
        normalise_inputs = hp.normalise_inputs
        normalise_targets = hp.normalise_targets
        self.x = np.float32(hp.data_x)
        self.y = np.float32(hp.data_y)
        self.t = np.float32(hp.data_t)

        if test:
            z = np.float32(hp.data_z)
            n_t = self.t.shape[0]
            xs, ys, ts, zs = [], [], [], []
            self.time_idx = []
            xx, yy = np.meshgrid(self.x, self.y)
            zs = np.transpose(z.copy(), (2, 0, 1)).reshape(-1, 1)
            for i in range(0, n_t):  # , n_t // 3):
                order = "C"
                xs.append(xx.reshape(-1, 1, order=order))
                ys.append(yy.reshape(-1, 1, order=order))
                ts.append(np.zeros_like(xs[-1]) + self.t[i])
                self.time_idx.append(i)

            xs = np.concatenate(xs, axis=0)
            ys = np.concatenate(ys, axis=0)
            ts = np.concatenate(ts, axis=0)
            samples = np.hstack([xs, ys, ts])
            targets = zs
        else:
            xytz = np.float32(hp.data_xytz)
            samples = xytz[:, :3]
            targets = xytz[:, 3:4]
        if normalise_inputs:
            nv_samples = self.normalize(samples, nv_samples, True)
        else:
            nv_samples = [(0, 1) for _ in range(self.input_size)]
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
            dtype = torch.float32
            device = "cuda"
        else:
            dtype = torch.bfloat32
            device = "cpu"

        self.samples = self.samples.to(device, dtype=dtype)
        if self.need_target:
            self.targets = self.targets.to(device, dtype=dtype)
        self.device = device
        self.dtype = dtype
