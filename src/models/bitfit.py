import torch
import torch.nn as nn
import re


def modify_with_bitfit(transformer, config):
    for m_name, modul