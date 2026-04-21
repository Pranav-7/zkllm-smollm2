"""
fileio_utils.py — Fixed-point I/O helpers for zkLLM.
Matches the exact semantics used throughout SMOLM2_135M_30_LAYERS.ipynb.
"""

import torch
import numpy as np


def save_int(tensor: torch.Tensor, scale: float, filename: str) -> None:
    """
    Multiply tensor by `scale`, round, cast to int32, write as raw binary.

    Usage:
        fileio_utils.save_int(X_padded, 1 << 16, 'layer0-input.bin')
        fileio_utils.save_int(rms_inv,  1 << 16, 'rms_inv_temp.bin')
        fileio_utils.save_int(ys,       1 << 16, 'swiglu-table.bin')
    """
    arr = torch.round(tensor.float() * scale).to(torch.int32)
    arr.cpu().numpy().astype(np.int32).tofile(filename)


def to_float(int_tensor: torch.Tensor, log_sf: int,
             dtype=None) -> torch.Tensor:
    """
    Divide integer tensor by 2^log_sf and return as float.

    Args:
        int_tensor : integer-valued torch tensor (int32 or int64)
        log_sf     : log2 of the scale factor
        dtype      : optional target dtype (e.g. torch.float64)

    Usage:
        Q = fileio_utils.to_float(torch.tensor(Q_int, device='cuda'), 16)
        v = fileio_utils.to_float(A, 20, torch.float64)
    """
    if dtype is None:
        return int_tensor.float() / (1 << log_sf)
    return int_tensor.to(dtype) / (1 << log_sf)


def to_int64(float_tensor: torch.Tensor, log_sf: int) -> torch.Tensor:
    """
    Multiply float tensor by 2^log_sf, round, return as int64.

    Usage:
        A     = fileio_utils.to_int64(A_,    VALUE_LOGSF)
        shift = fileio_utils.to_int64(shift, ACCU_LOGSF)
    """
    return torch.round(float_tensor * (1 << log_sf)).to(torch.int64)


def fromto_int64(float_tensor: torch.Tensor, log_sf: int) -> torch.Tensor:
    """
    Quantize to fixed-point and return as float (simulate integer round-trip).
    Equivalent to: round(tensor * 2^log_sf) / 2^log_sf

    Usage:
        attn_pre = fileio_utils.fromto_int64(attn_pre, VALUE_LOGSF)
    """
    scale = float(1 << log_sf)
    return torch.round(float_tensor * scale) / scale
