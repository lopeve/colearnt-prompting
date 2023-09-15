import torch
import torch.nn as nn


def sample_embed(embed, sample_size, start_idx, end_idx):
    embed_weight = embed.weight
    rand_idx = torch.randint(start_idx, end_idx, (sample_size,))
    return embed_weight[rand_idx].detach()


def get_embed_pad(embed, sample_size, start_idx, end_idx)