
import torch
import numpy as np
from pytorch_lightning import LightningDataModule
import datasets
from typing import Optional
from transformers import AutoTokenizer

class FinetuneDataModule(LightningDataModule):
    def __init__(self, config, tokenizer, dataset_reader):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_reader = dataset_reader

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        if self.config.few_shot:
            _ = self.dataset_reader.read_few_shot_dataset()

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        if self.config.few_shot:
            self.train_dataset = self.dataset_reader.read_few_shot_dataset()
        else:
            self.train_dataset = self.dataset_reader.read_orig_dataset("train")
        self.dev_dataset = self.dataset_reader.read_orig_dataset("validation")
        self.train_dataset = FinetuneDatasetWithTemplate(
            self.train_dataset, self.dataset_reader.get_train_template(), self.tokenizer
        )
        self.dev_dataset = FinetuneDatasetWithTemplate(
            self.dev_dataset, self.dataset_reader.get_eval_template(), self.tokenizer
        )
        print(f"Train size {len(self.train_dataset)}")
        print(f"Eval size {len(self.dev_dataset)}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=create_collate_fn(self.tokenizer.pad_token_id, pretrain=False),
            drop_last=True,
            num_workers=min([self.config.batch_size, self.config.num_workers]),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dev_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=create_collate_fn(self.tokenizer.pad_token_id, pretrain=False),
            num_workers=min([self.config.eval_batch_size, self.config.num_workers]),
        )


# for use *after* using a datasetreader to create a ds_dict
class CotrainDataModule(LightningDataModule):
    def __init__(self, config, tokenizer, ds_dict, dataset_reader):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.ds_dict = ds_dict
        self.dataset_reader = dataset_reader

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        self.train_dataset = self.ds_dict['train']
        self.dev_dataset = self.ds_dict['validation']
        if 'test' in self.ds_dict:
            self.test_dataset = self.ds_dict['test']
            self.test_dataset = FinetuneDatasetWithTemplate(
                self.test_dataset, self.dataset_reader.get_eval_template(), self.tokenizer
            )

        self.train_dataset = FinetuneDatasetWithTemplate(
            self.train_dataset, self.dataset_reader.get_train_template(), self.tokenizer
        )
        self.dev_dataset = FinetuneDatasetWithTemplate(
            self.dev_dataset, self.dataset_reader.get_eval_template(), self.tokenizer
        )

        print(f"Train size {len(self.train_dataset)}")
        print(f"Eval size {len(self.dev_dataset)}")
        print(f"Test size {len(self.test_dataset)}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=create_collate_fn(self.tokenizer.pad_token_id, pretrain=False),
            drop_last=True,
            num_workers=min([self.config.batch_size, self.config.num_workers]),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dev_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=create_collate_fn(self.tokenizer.pad_token_id, pretrain=False),
            num_workers=min([self.config.eval_batch_size, self.config.num_workers]),
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(