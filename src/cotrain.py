
import os
import gc
import torch
import argparse
import datasets
import logging
import numpy as np
import time
import copy
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from src.data import CotrainDataModule, get_dataset_reader, PretrainDataModule, BERTDataModule
from src.models.EncoderDecoder import EncoderDecoder
from src.models.BERT import BERT
from src.models.modify_model import modify_transformer
from src.utils.Config import Config
from src.utils.util import ParseKwargs, set_seeds
from src.utils.cotrain_utils import (
    get_conf_inds,
    get_conf_inds_per_class,
    get_dsdict_prompt,
    get_dsdict_bert,
)


def get_transformer(config, modify=True):
    print(config.origin_model)
    tokenizer = AutoTokenizer.from_pretrained(config.origin_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.origin_model, low_cpu_mem_usage=True)

    tokenizer.model_max_length = config.max_seq_len
    if modify:
        model = modify_transformer(model, config)
    return tokenizer, model




def main(config):
    """
    Trains the model

    :param config:
    :return:
    """


    """
    # todo: remove. for debuggin.
    ds_dict = dataset_reader.get_full_orig_dataset()
    print(ds_dict)
    bertname = 'bert-base-uncased'
    dm = BERTDataModule(bertname, 'cb', ds_dict)
    #dm.prepare_data()
    #dm.setup('fit')
    bert = BERT(
        model_name_or_path=bertname,
        num_labels=dm.num_labels,
        task_name=dm.task_name,
    )
    trainer = Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=1
    )
    trainer.fit(bert, datamodule=dm)
    ds_dict = get_dsdict_bert(
        config, args, dataset_reader, bert
    )
    # end todo remove
    """

    # in the prompt-tuning case, don't call modify() on
    # the model for the very first step because adding the noisy prompt
    # terms messes up the outputs.
    # in the paper we add [PAD] instead of doing this but that's not necessary.

    if config.model_modifier == 'prompt-tuning' and not config.prompt_tuning_init_with_pad:
        tokenizer, model = get_transformer(config, modify=False)
    else:
        tokenizer, model = get_transformer(config, modify=True)

    dataset_reader = get_dataset_reader(config)

    # this wrapper only uses the dataset reader for metrics
    model = EncoderDecoder(config, tokenizer, model, dataset_reader)

    original_exp_dir = config.exp_dir
    original_exp_name = config.exp_name
    original_beta = config.cotrain_beta

    for t in range(5):
        config.exp_dir = f'{original_exp_dir}_round{t+1}'
        config.exp_name = f'{original_exp_name}_round{t+1}'
        config.set_exp_dir() # calls mkdir

        # todo: make 0.1 a configurable amt
        config.cotrain_beta = original_beta + 0.1*t

        # get confidently-pseudolabeled data from T0
        ds_dict = get_dsdict_prompt(
            config, dataset_reader,
            tokenizer, model.to('cuda').model,
        )

        #import pdb
        #pdb.set_trace()
        del model

        gc.collect()
        torch.cuda.empty_cache()

        ### train BERT model ####
        ### then switch back to Prompt model ####

        # wrap up this pseudolabeled data in a data module for BERT
        dm = BERTDataModule(config.bert_name, config.dataset, ds_dict)

        bert = BERT(
            model_name_or_path=config.bert_name,
            num_labels=dm.num_labels,
            task_name=dm.task_name,
            dataset_reader=dataset_reader,