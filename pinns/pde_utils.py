from __future__ import annotations

from typing import Optional

import torch


def gen_uniform(
    bs: int,
    device: str | torch.device,
    start: float = -1,
    end: float = 1,
    temporal_scheme: bool = False,
    M: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate uniform random samples, optionally with temporal stratification.

    Parameters
    ----------
    bs : int
        Total number of samples (batch size).
    device : str or torch.device
        Device for the output tensor.
    start : float, optional
        Lower bound of the uniform distribution. Default: -1.
    end : float, optional
        Upper bound of the uniform distribution. Default: 1.
    temporal_scheme : bool, optional
        If True, use stratified sampling across M temporal bins. Default: False.
    M : int, optional
        Number of temporal bins (required if temporal_scheme=True).
    dtype : torch.dtype, optional
        Data type. Default: torch.float32.

    Returns
    -------
    torch.Tensor
        Random samples of shape (bs, 1) -- or (bs // M * M, 1) if
        temporal_scheme=True and bs is not divisible by M.
    """
    if temporal_scheme:
        step = (end - start) / M
        temporal_values = torch.arange(start, end + step, step=step)
        times = []
        for i in range(M):
            r1, r2 = temporal_values[i], temporal_values[i + 1]
            times.append(
                (r1 - r2)
                * torch.rand(
                    (bs // M, 1),
                    dtype=dtype,
                    device=device,
                    requires_grad=False,
                )
                + r2
            )
        vector = torch.cat(times, dim=0)
    else:
        vector = (end - start) * torch.rand(
            (bs, 1), dtype=dtype, device=device, requires_grad=False
        ) + start
    return vector
