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
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        dataset_reader,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.task_name = task_name
        print('loading model')
        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)

        if 'roberta' in model_name_or_path:
            module = self.model.roberta
            pooler = None
        elif 'deberta' in model_name_or_path:
            module = self.model.deberta
            pooler = self.model.pooler
        else:
            module = self.model.bert
            pooler = module.pooler

        # turn off everything (including embeddings)

        for param in module.parameters():
            param.requires_grad = False

        # comment these two blocks for linear-only
        # turn on the last layer and the pooler again.
        for param in module.encoder.layer[-1].parameters():
            param.requires_grad = True

        if pooler is not None:
            for param in pooler.parameters():
                param.requires_grad = True


        print('loaded model')
        self.dataset_reader = da