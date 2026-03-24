from __future__ import annotations

from typing import Optional

import torch
import numpy as np
from torch.utils.data import Dataset


class DataPlaceholder(Dataset):
    """Base dataset for PINN training with built-in normalisation and batching.

    Loads data from a NumPy .npy/.npz file, normalises inputs and targets
    to [-1, 1], and provides an iterator interface for mini-batch training.

    Parameters
    ----------
    path : str
        Path to a .npy file containing the data array.
    nv_samples : list of tuple, optional
        Pre-computed normalisation values for inputs as [(mean, scale), ...].
        If None, computed from data.
    nv_targets : list of tuple, optional
        Pre-computed normalisation values for targets.
    normalise_targets : bool, optional
        Whether to normalise target values. Default: True.
    gpu : bool, optional
        Whether to move data to GPU. Default: False.
    need_target : bool, optional
        Whether target values are present. Default: True.
    bs : int, optional
        Batch size. Default: 1.
    temporal_causality : bool, optional
        Whether to set up temporal causality bins. Default: False.
    M : int, optional
        Number of temporal bins for causality. Default: 32.

    Attributes
    ----------
    samples : torch.Tensor
        Normalised input coordinates.
    targets : torch.Tensor
        Normalised target values (if need_target=True).
    nv_samples : list of tuple
        Normalisation values for inputs: [(center, half_range), ...].
    nv_targets : list of tuple
        Normalisation values for targets.
    input_size : int
        Number of input dimensions.
    output_size : int
        Number of output dimensions.
    """

    def __init__(
        self,
        path: str,
        nv_samples: Optional[list[tuple[float, float]]] = None,
        nv_targets: Optional[list[tuple[float, float]]] = None,
        normalise_targets: bool = True,
        gpu: bool = False,
        need_target: bool = True,
        bs: int = 1,
        # temporal_causality=False,
        # M=32,
    ) -> None:
        self.need_target = need_target
        self.input_size = 3
        self.output_size = 1
        self.bs = bs
        self.test = False

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
        # if temporal_causality:
        #     self.temporal_causality = temporal_causality
        #     self.M = M
        #     self.setup_temporal_causality()

    def setup_data(self, pc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract samples and targets from raw loaded data.

        Override this in subclasses to define how raw numpy data is split
        into input coordinates and target values.

        Parameters
        ----------
        pc : np.ndarray
            Raw data loaded from the .npy file.

        Returns
        -------
        samples : np.ndarray
            Input coordinates, shape (n_samples, input_dims).
        targets : np.ndarray
            Target values, shape (n_samples, output_dims).
        """
        samples = pc[:, 0].astype(np.float32)
        targets = pc[:, 1].astype(np.float32)
        return samples, targets

    def setup_cuda(self, gpu: bool) -> None:
        """Move data to GPU/CPU and set dtype.

        Uses float16 on CUDA, bfloat16 on CPU.

        Parameters
        ----------
        gpu : bool
            Whether to use GPU.
        """
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

    def normalize(
        self,
        vector: np.ndarray,
        nv: Optional[list[tuple[float, float]]],
        include_last: bool = True,
    ) -> list[tuple[float, float]]:
        """Normalise columns of a 2D array to [-1, 1].

        Parameters
        ----------
        vector : np.ndarray
            2D array of shape (n_samples, n_features).
        nv : list of tuple or None
            Existing normalisation values. If None, computed from data.
        include_last : bool, optional
            Whether to normalise the last column. Default: True.

        Returns
        -------
        list of tuple
            Normalisation values [(center, half_range), ...].
        """
        c = vector.shape[1]
        if nv is None:
            nv = []
            for i, vect in enumerate(vector.T):
                if i == c - 1 and not include_last:
                    break
                m = (vect.max() + vect.min()) / 2
                s = (vect.max() - vect.min()) / 2
                if s == 0:
                    s = 1.0
                nv.append((m, s))

        for i in range(c):
            if i == c - 1 and not include_last:
                break
            vector[:, i] = (vector[:, i] - nv[i][0]) / nv[i][1]

        return nv

    def __len__(self) -> int:
        return self.samples.shape[0]

    def __getitem__(self, idx: torch.Tensor | int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        if not self.need_target:
            # return sample
            return {"x": sample}
        target = self.targets[idx]
        return {"x": sample, "z": target}
        # return sample, target

    def __next__(self) -> dict[str, torch.Tensor]:
        """Return the next mini-batch.

        For training (self.test=False): wraps around with re-shuffling
        when the epoch boundary is crossed.
        For testing (self.test=True): raises StopIteration at the end.

        Returns
        -------
        dict
            {'x': samples_batch} or {'x': samples_batch, 'z': targets_batch}.
        """
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

    def setup_batch_idx(self) -> None:
        """Initialise batch index ordering.

        Creates a random permutation for training or sequential index for testing.
        """
        self.last_idx = 0
        self.idx_max = self.__len__() // self.bs
        if not self.test:
            self.batch_idx = torch.randperm(self.__len__(), device=self.device)
        else:
            self.batch_idx = torch.arange(0, self.__len__(), device=self.device)

    # def setup_temporal_causality(self):
    #     """Quantise time values into M temporal bins for causal training.

    #     Stores quantised time indices in self.q_time.
    #     """
    #     time = self.samples[:, -1]
    #     start, end = time.min(), time.max()
    #     step = (end - start) / self.M
    #     self.q_time = torch.round(time / step) * step
