import torch.optim as optim
from transformers import Adafactor
import re
from collections import defaultdict


def get_optimizer(model, config):
    """
    Construct optimizer based on config

    :param model:
    :param config:
    :return:
    """
    optim_name = config.optimizer

    def param_name_to_group_name(param_name):
        if False:
            return ".".join(param_name.split(".")[:3])
            # only needed when the model has many trainable parameters, disabled in our expeirments
        else:
            return "."

    param_groups = defaultdict(lambda: {"params": []})
    trainable_param_names = set()
    for (param_name, param) in model.named_parameters():
        if re.fullmatch(config.trainable_param_names, param_name):
            param_gr