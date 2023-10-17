import torch.optim as optim
from transformers import Adafactor
import re
from collections import defaultdict


def get_optimizer(model,