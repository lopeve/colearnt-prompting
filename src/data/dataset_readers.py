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
    "hotpot_qa_fullwiki_Generate_Answer",
    "hotpot_qa_distractor_Generate_Answer",
    "hotpot_qa_distractor_Generate_Title_2",
    "hotpot_qa_fullwiki_Generate_Title_2",
    "hotpot_qa_fullwiki_Generate_Title_1",
    "hotpot_qa_distractor_Generate_Title_1",
    "hotpot_qa_distractor_Generate_Question",
    "hotpot_qa_fullwiki_Generate_Question",
    "tab_fact_tab_fact_tab_fact_3",
    "tab_fact_tab_fact_tab_fact_2",
    "tab_fact_tab_fact_tab_fact_1",
    "tab_fact_tab_fact_tab_fact_7",
    "tab_fact_tab_fact_tab_fact_4",
    "tab_fact_tab_fact_tab_fact_5",
    "tab_fact_tab_fact_tab_fact_6",
    "wiki_hop_masked_Choose_Best_Object_Candidate",
    "wiki_hop_masked_Indirect_Question_about_Birthplace_Citizenship_Place_of_Death",
    "narrativeqa_Template_05",
    "ecthr_cases_alleged_violation_prediction_silver_rationales",
    # "amazon_polarity/amazon_polarity",
    # "quail_context_question_answer_description_id",
    # "quail_context_question_description_answer_text",
    # "quail_context_question_answer_description_text",
    # "quail_context_question_description_answer_id",
    # "quail_context_question_answer_description_id",
    # "quail_context_question_description_answer_text",
    # "quail_context_question_answer_description_text",
    # "quail_context_question_description_answer_id",
    # "quail_description_context_question_text",
    # "quail_description_context_question_answer_text",
    # 'quail_context_description_question_answer_id',
    # 'quail_context_description_question_answer_text',
    # 'quail_context_description_question_text',
    # 'quail_context_question_answer_description_text',
    # 'quail_context_question_description_answer_id',
    # 'quail_context_question_description_text',
    # 'quail_description_context_question_answer_id',
    # 'quail_description_context_question_answer_text',
    # 'quail_description_context_question_text',
    # 'quail_no_prompt_id',
    # 'quail_no_prompt_text',
    # Tasks with broken cached files
    "gigaword_summarize_",
]


class BaseDatasetReader(object):
    """
    DatasetReader is responsible for reading and processing dataset
    """

    def __init__(self, config, dataset_stash):
        """
        :param config:
        """
        self.config = config
        self.dataset_stash = dataset_stash

        # todo: what does this do on new dataset?
        self.templates = DatasetTemplates(*self.dataset_stash)
        self.train_template = self.get_template(self.config.train_template_idx)
        self.eval_template = self.get_template(self.config.eval_template_idx)

    def get_template(self, template_idx):
        template_names = self.templates.all_template_names
        if template_idx >= 0:
            return self.templates[template_names[template_idx]]
        elif template_idx == -1: # random template choice in each step

            list_idx = []
            list_templates = []
            for idx, template_name in enumerate(template_names):
                if self.templates[template_name].metadata.original_task:
                    list_idx.append(idx)
                    list_templates.append(self.templates[template_name])
            print(list_idx)

            return list_templates
        elif template_idx == -2:
            return [self.templates[template_name] for template_name in template_names]

    def get_train_template(self):
        return self.train_template

    def get_eval_template(self):
        return self.eval_template

    def get_full_orig_dataset(self):
        if (self.config.local_path is not None) and os.path.exists(self.config.local_path):
            print(f"loading split from {self.config.local_path}")
            orig_data = load_from_disk(self.config.local_path)
        elif os.path.exists(DATASETS_OFFLINE):
            orig_data = load_from_disk(os.path.join(DATASETS_OFFLINE, *self.dataset_stash))
        else:
            orig_data = load_dataset(*self.dataset_stash)
        return orig_data

    def read_orig_dataset(self, split):
        """
        Read the original dataset

        :param split: split of data
        """
        if (self.config.local_path is not None) and os.path.exists(self.config.local_path):
            print(f"loading split from {self.config.local_path}")
            orig_data = load_from_disk(self.config.local_path)[split]
        elif os.path.exists(DATASETS_OFFLINE):
            orig_data = load_from_disk(os.path.join(DATASETS_OFFLINE, *self.dataset_stash))[split]
        else:
            #proxy = {'http': 'socks5://localhost:9000', 'https': 'socks5://localhost:9000'}
            #dl_conf = DownloadConfig(proxies=proxy, )
            orig_data = load_dataset(*self.dataset_stash, split=split)
            #, cache_dir="/home/hlang/hf_home/datasets")#download_config=dl_conf)#, cache_dir=os.environ["HF_HOME"])#, download_config=dl_conf)
        return orig_data

    def read_few_shot_dataset(self):
        file_dir = os.path.join("data", "few_shot", self.config.dataset, f"{self.config.num_shot}_shot")
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        file_path = os.path.join(file_dir, f"{self.config.few_shot_random_seed}_seed.jsonl")

        if os.path.exists(file_path):
            with open(file_path, "r") as fin:
                data = []
                for idx, line in enumerate(fin.readlines()):
                    data.append(json.loads(line.strip("\n")))

            return data
        else:
            orig_data = self.read_orig_dataset("train")
            selected_data = self._sample_few_shot_data(orig_data)

            with open(file_path, "w+") as fout:
                for example in selected_data:
                    fout.write(json.dumps(example) + "\n")
            return selected_data

    def _sample_few_shot_data(self, orig_data):
        saved_random_state = np.random.get_state()
        np.random.seed(self.config.few_shot_random_seed)
        orig_data = [x for x in orig_data]
        np.random.shuffle(orig_data)
        selected_data = orig_data[: self.config.num_shot]
        np.random.set_state(saved_random_state)
        return selected_data

    def compute_metric(self, accumulated):
        matching = [a == b for a, b in zip(accumulated["prediction"], accumulated["label"])]
        accuracy = sum(matching) / len(matching)
        bal_acc = balanced_accuracy_score(accumulated["label"], accumulated["prediction"])
        return {"accuracy": accuracy, "balanced_accuracy": bal_acc}


class StoryClozeReader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("story_cloze", "2016"))

    def read_orig_dataset(self, split):
        if split == "train":
            split = "validation"
        elif split == "validation":
            split = "test"

        if os.path.exists(DATASETS_OFFLINE):
            orig_data = load_from_disk(os.path.join(DATASETS_OFFLINE, *self.dataset_stash))[split]
        else:
            orig_data = load_dataset(
                *self.dataset_stash, split=split, data_dir="/fruitbasket/datasets/hugging_face/story_cloze"
            )
        orig_data = [example for example in orig_data]
        for idx, example in enumerate(orig_data):
            example["label"] = example["answer_right_ending"] - 1
            example["idx"] = idx
        return orig_data


class ANLIR1Reader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("anli",))

    