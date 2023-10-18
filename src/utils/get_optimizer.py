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
    optim_name 