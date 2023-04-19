
import torch
import torch.nn as nn
import re
from .AdapterVariants.Adapters import Adapter, LowRankAdapter, HyperComplexAdapter


def get_adapter(adapter_type):
    if adapter_type == "normal":
        return Adapter
    elif adapter_type == "lowrank":
        return LowRankAdapter
    elif adapter_type == "compacter":
        return HyperComplexAdapter
    else:
        raise ValueError("Not Implemented")


class T5LayerFFWithAdapter(nn.Module):
    def __init__(self, T5LayerFF, config, transformer_config):
        super().__init__()
        self.DenseReluDense = T5LayerFF.DenseReluDense
        self.adapter = get_adapter(config.adapter_type)(config, transformer_config)
        self.layer_norm = T5LayerFF.layer_norm
        self.dropout = T5LayerFF.dropout

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        adapter_output = self.adapter(forwarded_states)
        hidden_states = hidden_states + self.dropout(adapter_output)
        return hidden_states


class T5LayerSelfAttentionWithAdapter(nn.Module):
    def __init__(self, T5LayerSelfAttention, config, transformer_config):
        super().__init__()
        self.SelfAttention = T5LayerSelfAttention.SelfAttention
        self.adapter = get_adapter(config.adapter_type)(config, transformer_config)
        self.layer_norm = T5LayerSelfAttention.layer_norm
        self.dropout = T5LayerSelfAttention.dropout