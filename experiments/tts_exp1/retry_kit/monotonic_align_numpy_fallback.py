"""Pure-numpy monotonic alignment search (replaces the Cython ext for
local export on machines without a C compiler).

Equivalent to the standard VITS maximum_path (same DP recurrence and
backtracking as the Cython core).
"""
import numpy as np
import torch


def maximum_path(neg_cent, mask):
    """neg_cent: [b, t_t, t_s], mask: [b, t_t, t_s] (0/1)."""
    device = neg_cent.device
    dtype = neg_cent.dtype

    neg = neg_cent.data.cpu().numpy().astype(np.float32)
    valid = mask.data.cpu().numpy().astype(bool)

    b, t_t, t_s = neg.shape
    value = np.full((b, t_t, t_s), -1e30, dtype=np.float32)

    for n in range(b):
        for t in range(t_t):
            for s in range(t_s):
                if not valid[n, t, s]:
                    continue
                if t == 0 and s == 0:
                    value[n, t, s] = neg[n, t, s]
                elif t == 0:
                    value[n, t, s] = value[n, t, s - 1] + neg[n, t, s]
                elif s == 0:
                    value[n, t, s] = value[n, t - 1, s] + neg[n, t, s]
                else:
                    value[n, t, s] = (
                        max(value[n, t - 1, s], value[n, t, s - 1])
                        + neg[n, t, s]
                    )

    path = np.zeros((b, t_t, t_s), dtype=np.float32)
    for n in range(b):
        # last valid column
        row_t = t_t - 1
        last = np.where(valid[n, row_t])[0]
        if last.size == 0:
            continue
        s = int(last[-1])
        for t in range(row_t, -1, -1):
            if s < 0:
                break
            path[n, t, s] = 1.0
            if t == 0 and s == 0:
                break
            if t == 0:
                s -= 1
            elif s == 0:
                t -= 1
            else:
                if value[n, t - 1, s] > value[n, t, s - 1]:
                    t -= 1
                else:
                    s -= 1

    return torch.from_numpy(path).to(device=device, dtype=dtype)
