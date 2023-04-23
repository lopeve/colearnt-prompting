import torch
import torch.nn as nn
import re


def modify_with_bitfit(transformer, config):
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config.bitfit_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.bitfit_layers, c_name):
                    layer.bias = nn.Parameter(torch.zeros(layer.out_features))
    return transf