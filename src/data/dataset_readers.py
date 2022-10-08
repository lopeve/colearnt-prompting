import os
import json
import numpy as np
from datasets import load_dataset, load_from_disk, DownloadConfig
from promptsource.templates import DatasetTemplates
import pkg_resources
from promptsource import templates
import csv
from typing import Dict, List, Optional, Tuple
import re
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

def get_dataset_reader(config):
    dataset_class = {
        "T0Mixture": T0MixtureReader,
        "rte": RTEReader,
        "boolq": BoolQReader,
        "h-swag": HSwagReader,
        "copa": COPAReader,
        "wic": WiCReader,
        "winogrande": WinograndeReader,
        "cb": CBReader,
        "storycloze": StoryClozeReader,
        "anli-r1": ANLIR1Reader,
        "anli-r2": ANLIR2Reader,
        "anli-r3": ANLIR3Reader,
        "wsc": WSCFixedReader,
        "ade_corpus_v2": RaftReader,
        "banking_77": RaftReader,
        "terms_of_service": RaftReader,
        "tai_safety_research": RaftReader,
        "neurips_impact_statement_risks": RaftReader,
        "overruling": RaftReader,
        "systematic_review_inclusion": RaftReader,
        "one_stop_english": RaftReader,
        "tweet_eval_hate": RaftReader,
        "twitter_complaints": RaftReader,
        "semiconductor_org_types": RaftReader,
        "gpt-rte": GPTReader,
        "gpt-cb": GPTReader,
        "gpt-trec": GPTReader
    }[config.dataset]
    return dataset_class(config)


DATASETS_OFFLINE = "/home/hlang/datasets_offline/"
GPT_DATA_ROOT="./gpt-data"

MAX_EXAMPLES_PER_DATASET = 500_000
TASK_BLACKLIST = [
    # Tasks which often tokenize to > 1024 tokens currently
    "hotpot_qa_distractor_Generate_Explanations",
    "hotpot_qa_fullwiki_Generate_Explanations",
    "hotpot_qa_distractor_Generate_Answer_and_Explanations",
    "hotpot_qa_fullwiki_Generate_Answer_and_Explanations",
    "hotpo