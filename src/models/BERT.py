import os
import json
import torch
import torch.nn.functional as F
import torch.distributed as dist
from pytorch_lightning import LightningModule
from typing import Optional
from torch.optim import AdamW
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
import datasets
from datetime import datetime

# taken and modified from lightning docs
class BERT(LightningModule):
    def __init_