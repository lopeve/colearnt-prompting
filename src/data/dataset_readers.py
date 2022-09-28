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
    dataset_class = 