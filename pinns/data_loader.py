import torch
import numpy as np

from torch.utils.data import Dataset


class DataPlaceholder(Dataset):
    def __init__(
        self,
        path,
        nv_samples=None,
        nv_targets=None,
        normalise_targets=True,
        gpu=False,
        need_target=True,
        bs=1,
        temporal_causality=False,
        M=32,
    ):
        self.need_target = need_target
        self.input_size = 3
        self.output_size = 1
        self.bs = bs

        pc = np.load(path)
        samples, targets = self.setup_data(pc)
        nv_samples = self.normalize(samples, nv_samples, True)
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
        if temporal_causality:
            self.temporal_causality = temporal_causality
            self.M = M
            self.setup_temporal_causality()

    def setup_data(self, pc):
        samples = pc[:, 0].astype(np.float32)
        targets = pc[:, 1].astype(np.float32)
        return samples, targets

    def setup_cuda(self, gpu):
        if gpu:
            dtype = torch.float16
            device = "cuda"
        else:
            dtype = torch.bfloat16
            device = "cpu"

        self.samples = self.samples.to(device, dtype=dtype)
        if self.need_target:
            self.targets = self.targets.to(device, dtype=dtype)
        self.device = device
        self.dtype = dtype

    def normalize(self, vector, nv, include_last=True):
        c = vector.shape[1]
        if nv is None:
            nv = []
            for i, vect in enumerate(vector.T):
                if i == c - 1 and not include_last:
                    break
                m = (vect.max() + vect.min()) / 2
                s = (vect.max() - vect.min()) / 2
                nv.append((m, s))

        for i in range(c):
            if i == c - 1 and not include_last:
                break
            vector[:, i] = (vector[:, i] - nv[i][0]) / nv[i][1]

        return nv

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if not self.need_target:
            # return sample
            return {"x": sample}
        target = self.targets[idx]
        return {"x": sample, "z": target}
        # return sample, target

    def __next__(self):
        if self.last_idx + self.bs <= self.__len__():
            idx_bs = self.batch_idx[self.last_idx : (self.last_idx + self.bs)]
            self.last_idx += self.bs
            return self.__getitem__(idx_bs)
        else:
            if self.test:
                raise StopIteration
            else:
                end_of_batch = self.batch_idx[self.last_idx :]
                rest = self.bs - end_of_batch.shape[0]
                self.batch_idx = torch.randperm(self.__len__(), device=self.device)
                idx_bs = torch.cat([end_of_batch, self.batch_idx[:rest]])
                self.last_idx = rest
                return self.__getitem__(idx_bs)

    def setup_batch_idx(self):
        self.last_idx = 0
        self.idx_max = self.__len__() // self.bs
        if not self.test:
            self.batch_idx = torch.randperm(self.__len__(), device=self.device)
        else:
            self.batch_idx = torch.arange(0, self.__len__(), device=self.device)

    def setup_temporal_causality(self):
        time = self.samples[:, -1]
        start, end = time.min(), time.max()
        step = (end - start) / self.M
        self.q_time = torch.round(time / step) * step

        # import pdb; pdb.set_trace()
        # temporal_values = torch.arange(start, end + step, step=step)
