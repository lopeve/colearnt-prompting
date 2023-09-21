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
        self.num_prefix_emb = config.prompt_tuning_num_prefix_emb
        self.prompt_tuning_encoder = config.prompt_tuning_encoder
        self.encoder = encoder

        embed_sample_fn = get_embed_pad if config.prompt_tuning_init_with_pad else sample_embed

        self.prompt_embedding = nn.Parameter(
            embed_sample_fn(
                embed=encoder.get_input_embeddings(),
                sample_size=self.num_prefix_emb,
                start_idx=3,
                end_idx=5003,
            )
        )  # [num_prefix_emb, emb_dim] sampled from 5000 most common regular token embeddings

    def forward(self, input_ids, attention_mask, inputs_em