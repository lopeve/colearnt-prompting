import torch
import torch.nn as nn


def sample_embed(embed, sample_size, start_idx, end_idx):
    embed_weight = emb