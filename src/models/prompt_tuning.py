import torch
import torch.nn as nn


def sample_embed(embed, sample_size, start_idx, end_idx):
    embed_weight = embed.weight
    rand_idx = torch.randint(start_idx, end_idx, (sample_size,))
    return embed_weight[rand_idx].detach()


def get_embed_pad(embed, sample_size, start_idx, end_idx):
    embed_weight = embed.weight
    pad_idx = torch.zeros((sample_size,)).long() # pad token is idx 0
    return embed_weight[pad_idx].detach()

class T5EncoderPromptTuningWrapper(nn.Module):
    def __init__(self, encoder, config):
        super().__init__()
        self.num_prefix_emb = config.prom