# The codes are from Armen Aghajanyan from facebook, from paper
# Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning
# https://arxiv.org/abs/2012.13255

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Set
from .fwh_cuda import fast_walsh_hadamard_transform as fast_walsh_hadamard_transform_cuda


def fast_walsh_hadamard_torched(x, axis: int = 0, normalize: bool = True):
    orig_shape = x.size()
    assert axis >= 0 and axis < len(orig_shape), "For a vector of shape %s, axis must be in [0, %d] but it is %d" % (
        orig_shape,
        len(orig_shape) - 1,
        axis,
    )
    h_dim = orig_shape[axis]
    h_dim_exp = int(round(np.log(h_dim) / np.log(2)))
    assert h_dim == 2 ** h_dim_exp, (
        "hadamard can only be computed over axis with size that is a power of two, but"
        " chosen axis %d has size %d" % (axis, h_dim)
    )

    working_shape_pre = [int(torch.prod(torch.tensor(orig_shape[:axis])))]
    working_shape_post = [int(torch.prod(torch.tensor(orig_shape[axis + 1 :])))]
    working_shape_mid = [2] * h_dim_exp
    working_shape = working_shape_pre + working_sh