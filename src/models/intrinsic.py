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
    working_shape = working_shape_pre + working_shape_mid + working_shape_post

    ret = x.view(working_shape)

    for ii in range(h_dim_exp):
        dim = ii + 1
        arrs = torch.chunk(ret, 2, dim=dim)
        assert len(arrs) == 2
        ret = torch.cat((arrs[0] + arrs[1], arrs[0] - arrs[1]), axis=dim)

    if normalize:
        ret = ret / np.sqrt(float(h_dim))

    ret = ret.view(orig_shape)

    return ret


def fastfood_vars(DD, device=0):
    """
    Returns parameters for fast food transform
    :param DD: desired dimension
    :return:
    """
    ll = int(np.ceil(np.log(DD) / np.log(2)))
    LL = 2 ** ll

    # Binary scaling matrix where $B_{i,i} \in \{\pm 1 \}$ drawn iid
    BB = torch.FloatTensor(LL).uniform_(0, 2).type(torch.LongTensor)
    BB = BB * 2 - 1
    BB.requires_grad_(False)

    # Random permutation matrix
    Pi = torch.LongTensor(np.random.permutation(LL))
    Pi.requires_grad_(False)

    # Gaussian scaling matrix, whose elements $G_{i,i} \sim \mathcal{N}(0, 1)$
    GG = torch.FloatTensor(
        LL,
    ).normal_()
    GG.requires_grad_(False)
    divisor = torch.sqrt(LL * torch.sum(torch.pow(GG, 2)))
    return [BB.to(device), Pi.to(device), GG.to(device), divisor.to(device), LL]


def random_vars(desired_dim, intrinsic_dim, device=0):
    """Returns a random matrix of the desired dimension."""
    R = torch.FloatTensor(desired_dim, intrinsic_dim).normal_(std=0.01).to(device)
    R.requires_grad_(False)
    divisor = torch.norm(R)
    return [R, divisor]


def fastfood_torched(x, DD: int, param_list: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]):
    """
    Fastfood transform
    :param x: array of dd dimension
    :param DD: desired dimension
    :return:
    """
    dd = x.size(0)

    BB, Pi, GG, divisor, LL = param_list
    # Padd x if needed
    dd_pad = F.pad(x, pad=(0, LL - dd), value=0.0, mode="constant")
    # From left to right HGPiH(BX), where H is Walsh-Hadamard matrix
    dd_pad = dd_pad * BB

    # HGPi(HBX)
    mul_2 = FastWalshHadamard.apply(dd_pad)

    # HG(PiHBX)
    mul_3 = mul_2[Pi]

    # H(GPiHBX)
    mul_3 = mul_3 * GG

    # (HGPiHBX)
    mul_5 = FastWalshHadamard.apply(mul_3)

    ret = mul_5[: int(DD)]
    ret = ret / (divisor * np.sqrt(float(DD) / LL))
    return ret


def random_torched(intrinsic_vec, param_list: Tuple[torch.Tensor, int]):
    """Random dense transform"""
    R, divisor = param_list
    result = torch.matmul(R, intrinsic_vec)
    # TODO: for now we are not normalizing with the divisor, to be added later.
    return result


class FastWalshHadamard(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(torch.tensor([1 / np.sqrt(float(input.size(0)))]).to(input))
        if input.is_cuda:
            return fast_walsh_hadamard_transform_cuda(input.float(), False)
        else:
            return fast_walsh_hadamard_torched(input.float(), normalize=False)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        if grad_output.is_cuda:
            return input * fast_walsh_hadamard_transform_cuda(grad_output.clone().float(), False).to(grad_output)
        else:
            return input * fast_walsh_hadamard_torched(grad_output.clone().float(), normalize=False).to(grad_output)


class IntrinsicDimensionLight:
    def __init__(
        self,
        module: nn.Module,
        intrinsic_dimension: int,
        str_filter: Set[str] = set(),
        said=False,
        projection="fastfood",
        device="cpu",
    ):
        """
        Adds hook only for the parameters selected inside the str_filter, and if str_filter is empty, this selects
        all the parameters with gradient = True.
        """
        self.projection = projection
        self.name_base_localname = []
        self.initial_value = dict()
        self.projection_params = {}
        self.said = said
        self.device = device
        self.said_size = len(list(module.named_parameters()))
        if self.said:
            assert intrinsic_dimension > self.said_size
            intrinsic_dimension -= self.said_size + 1

        length = 0
        for name, param in module.named_parameters():
            if param.requires_grad and (len(str_filter) == 0 or any([x in name for x in str_filter])):
                length += 1
                self.initial_value[name] = v0 = param.clone().detach().requires_grad_(False).to(self.device)
                # DD = np.prod(v0.size())
                DD = np.sum(v0.size())
                self.projection_params[name] = self.get_projection_params(DD, self.device)
                base, localname = module, name
                while "." in localname:
                    prefix, localname = localname.split(".", 1)
                    base = base.__getattr__(prefix)
                self.name_base_localname.append((name, base, localname))
                param.requires_grad_(False)

        self.intrinsic_dimension = intrinsic_dimension
        self.intrinsic_parameter = nn.Parameter(torch.zeros((intrinsic_dimension), device=self.device))
        module.intrinsic_parameter = self.intrinsic_parameter
        # module.register_parameter("intrinsic_parameter", self.intrinsic_parameter)
        # setattr(module, "intrinsic_parameter", self.intrinsic_parameter)

        if said:
            self.intrinsic_parameter_said = nn.Parameter(torch.ones((length), device=self.device))
            module.intrinsic_parameter_said = self.intrinsic_parameter_said
            # module.register_parameter("intrinsic_parameter_said", self.intrinsic_parameter_said)
            # setattr(module, "intrinsic_parameter_said", self.intrinsic_parameter_said)

    def get_projection_params(self, DD, device):
        if self.projection == "fastfood":
            return fastfood_vars(DD, device)
        elif self.projection == "random":
            return random_vars(DD, self.intrinsic_dimension, device)

    def move_to(self, x_tuple, target):
        if isinstance(x_tuple, torc