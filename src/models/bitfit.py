import torch
import torch.nn as nn
import re


def modify_with_bitfit(transformer, config):
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config.bitfit_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.bitfit_layers, c_name):
                    layer.bias = nn.Parameter(torch.zeros(layer.out_features))
    return transformer


if __name__ == "__main__":
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    class BitFitConfig:
        def __init__(self):
            self.bitfit_modules = ".*"
            self.bitfit_layers = "q|k|v|o|w.*"
            self.trainable_param_names = ".*layer_norm.*|.*bias"
            # lora_modules and lora_layers are speicified with regular expressions
            # see https://www.w3schools.com/python/python_regex.asp for reference

   