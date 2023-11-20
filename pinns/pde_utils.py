import torch


def gen_uniform(
    bs, device, start=-1, end=1, temporal_scheme=False, M=None, dtype=torch.float32
):
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
        vector = torch.cat(times, axis=0)
    else:
        vector = (end - start) * torch.rand(
            (bs, 1), dtype=dtype, device=device, requires_grad=False
        ) + start
    return vector
