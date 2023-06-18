# The codes are from Armen Aghajanyan from facebook, from paper
# Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning
# https://arxiv.org/abs/2012.13255

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Set
from .fwh_cuda import fast_walsh_hadamard_transform as fast_walsh_hadamard_transform_cuda


def fast_walsh_hadamard_torched(x, axis: int = 0, normal