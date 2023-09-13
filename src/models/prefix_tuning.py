
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
from src.utils.get_optimizer import get_optimizer


class T5AttentionPrefixTuning(nn.Module):
    def __init__(self, attention_layer, num_prefix_tokens, parameterization, shared=None):
        super().__init__()
        self.is_decoder = attention_layer.is_decoder
        self.has_relative_attention_bias = attention_layer.has_relative_attention_bias

        self.relative_attention_num_buckets = attention_layer.relative_attention_num_buckets
        self.d_model = attention_layer.d_model
        self.key_value_proj_dim = attention_layer.key_value_proj_dim
        self.n_heads = attention_layer.n_heads
        self.dropout = attention_layer.dropout
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        self.prune_heads = attention_layer.prune_heads
        self._relative_position_bucket = attention_layer._relative_position_bucket
        self.compute_bias = attention_layer.compute_bias

        self.q = attention_layer.q
        self.k = attention_layer.k
        self.v = attention_layer.v
        self.o = attention_layer.o
        if self.has_relative_attention_bias:
            self.relative_attention_bias = attention_layer.relative_attention_bias
        self.pruned_heads = attention_layer.pruned_heads
        self.gradient_checkpointing = attention_layer.gradient_checkpointing

        self.parameterization = parameterization
        self.num_prefix_tokens = num_prefix_tokens
        self.mode = "apply"

        self.setup_prefix(shared)

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Modified from T5Attention forward
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        pask_key_value, query_length, use_cache disabled
        """
        assert past_key_value is None
        assert query_length is None
        assert not use_cache
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]
        key_length = seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, prefix_states):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                output_states = proj_layer(hidden_states)
            else:
                # cross-attn
                output_states = proj_layer(key_value_states)
            if prefix_states is not None:
                output_states = torch.cat([prefix_states, output_states], dim=1)
            return output_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        if self.mode == "apply":
            prefix = self.get_prefix(batch_size)
            key_length += self.num_prefix_tokens
        else:
            prefix = (None, None)

        key_states = project(hidden_states, self.k, key_value_states, prefix[0])
        value_states = project(hidden_states, self.v, key_value_states, prefix[1])

        if self.mode == "store":
            self.stored_key_value_states = (key_states, value_states)

        key_states, value_states = shape(key_states), shape(value_states)

        # compute scores
        scores = torch.matmul(