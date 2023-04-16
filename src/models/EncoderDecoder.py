
import os
import json
import torch
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from pytorch_lightning import LightningModule
from src.utils.get_optimizer import get_optimizer
from src.utils.get_scheduler import get_scheduler
from statistics import mean
from deepspeed.utils import zero_to_fp32
from .fishmask import fishmask_plugin_on_init, fishmask_plugin_on_optimizer_step, fishmask_plugin_on_end


class EncoderDecoder(LightningModule):
    """
    Encoder Decoder
    """

    def __init__(self, config, tokenizer, transformer, dataset_reader, track_metric=None, mode='max'):
        """
        :param config
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.model = transformer
        self.dataset_reader = dataset_reader

        self.use_deepspeed = self.config.compute_strategy.startswith("deepspeed")
        self.use_ddp = self.config.compute_strategy.startswith("ddp")
        self.load_model()

        # need to keep track of this custom since default lightning checkpointing
        # seems tricky to get the custom saving code
        self.track_metric = track_metric
        self.comparator = np.greater if mode == 'max' else np.less
        self.best_metric_val = None

        self._last_global_step_saved = -1

        if self.config.fishmask_mode is not None:
            fishmask_plugin_on_init(self)

    def training_step(self, batch, batch_idx):
        if self.config.model_modifier == "intrinsic":
            from .intrinsic import intrinsic_plugin_on_step
            intrinsic_plugin_on_step(self)

        if self.config.mc_loss > 0 or self.config.unlikely_loss > 0:
            input_ids, choices_ids, labels = batch["input_ids"], batch["answer_choices_ids"], batch["labels"]
            bs, num_choices = choices_ids.size()[:2]

            flat_choices_ids = choices_ids.flatten(0, 1)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).float()  # [bs, max_seq_len]
            encoder_hidden_states = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
            encoder_hidden_states = encoder_hidden_states.unsqueeze(dim=1).repeat(1, num_choices, 1, 1).flatten(0, 1)
            attention_mask = attention_mask.unsqueeze(dim=1).repeat(1, num_choices, 1).flatten(0, 1)
            decoder_input_ids = torch.cat([torch.zeros_like(flat_choices_ids[:, :1]), flat_choices_ids[:, :-1]], dim=1)
            decoder_attention_mask = (decoder_input_ids == decoder_input_ids).float()
            lm_target = flat_choices_ids - 100 * (flat_choices_ids == self.tokenizer.pad_token_id).long()

            model_output = self.model(
                attention_mask=attention_mask,
                encoder_outputs=[encoder_hidden_states],
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )
            choices_scores = (
                F.cross_entropy(model_output.logits.flatten(0, 1), lm_target.flatten(0, 1), reduction="none")
                .view(bs, num_choices, -1)
                .sum(dim=-1)
            )
            if self.config.length_norm > 0:
                choices_scores = choices_scores / torch.pow(
                    (choices_ids != self.tokenizer.pad_token_id).sum(dim=-1), self.config.length_norm
                )
            lm_loss = F.cross_entropy(
                model_output.logits.view(bs, num_choices, *model_output.logits.size()[1:])[range(bs), labels].flatten(
                    0, 1
                ),
                lm_target.view(bs, num_choices, -1)[range(bs), labels].flatten(0, 1),
            )

            tensorboard_logs = {"lm_loss": lm_loss.item()}
            if self.config.mc_loss > 0:
                mc_loss = F.cross_entropy(-choices_scores, labels)
                tensorboard_logs["mc_loss"] = mc_loss.item()
            else:
                mc_loss = 0.0

            if self.config.unlikely_loss > 0:
                cand_loglikely = -F.cross_entropy(
                    model_output.logits.flatten(0, 1), lm_target.flatten(0, 1), reduction="none"
                ).view(bs, num_choices, -1)
                cand_loglikely += (lm_target < 0).view(bs, num_choices, -1) * -100
                cand_loglikely[range(bs), labels] = -100
                unlikely_loss = -torch.log(1 - torch.exp(cand_loglikely) + 1e-2).sum() / (cand_loglikely != -100).sum()
                tensorboard_logs["unlikely_loss"] = unlikely_loss.item()
            else:
                unlikely_loss = 0.0

            loss = lm_loss + mc_loss * self.config.mc_loss + unlikely_loss * self.config.unlikely_loss
            tensorboard_logs["loss"] = loss.item()
        else:
            input_ids, target_ids = batch["input_ids"], batch["target_ids"]
            attention_mask = (input_ids != self.tokenizer.pad_token_id).float()  # [bs, max_seq_len]
            lm_labels = target_ids + -100 * (target_ids == self.tokenizer.pad_token_id).long()  # [bs, max_seq_len]
            decoder_input_ids = torch.cat(
                [torch.zeros_like(lm_labels[:, :1]), target_ids[:, :-1]], dim=1
            )  # [bs, max_seq_len]
            decoder_attention_mask = (decoder_input_ids == decoder_input_ids).float()

            model_output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=lm_labels,
            )
            loss = model_output.loss
            tensorboard_logs = {"loss": loss.item()}

        if not (self.use_deepspeed or self.use_ddp) or dist.get_rank() == 0:
            self.log_dict(tensorboard_logs)

        if self.global_step % self.config.save_step_interval == 0:
            self.save_model()

        return loss


    def predict_step(self, batch, batch_idx):
        return self.predict(batch)

    def predict(self, batch):
        """
        Predict the lbl for particular pet
        :param batch:
        :param pet:
        :return:
        """
        if self.config.model_modifier == "intrinsic":
            intrinsic_plugin_on_step(self)

        input_ids, choices_ids, labels = batch["input_ids"], batch["answer_choices_ids"], batch["labels"]

        if not self.config.split_option_at_inference:
            bs, num_choices = choices_ids.size()[:2]
            flat_choices_ids = choices_ids.flatten(0, 1)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).float()  # [bs, max_seq_len]
            encoder_hidden_states = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
            encoder_hidden_states = encoder_hidden_states.unsqueeze(dim=1).repeat(1, num_choices, 1, 1).flatten(0, 1)
            attention_mask = attention_mask.unsqueeze(dim=1).repeat(1, num_choices, 1).flatten(0, 1)
            decoder_input_ids = torch.cat([torch.zeros_like(flat_choices_ids[:, :1]), flat_choices_ids[:, :-1]], dim=1)
            decoder_attention_mask = (decoder_input_ids == decoder_input_ids).float()
            lm_target = flat_choices_ids - 100 * (flat_choices_ids == self.tokenizer.pad_token_id).long()

            model_output = self.model(
                attention_mask=attention_mask,
                encoder_outputs=[encoder_hidden_states],
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )
            choices_scores = (
                F.cross_entropy(model_output.logits.flatten(0, 1), lm_target.flatten(0, 1), reduction="none")
                .view(bs, num_choices, -1)
                .sum(dim=-1)
            )
            if self.config.length_norm > 0:
                choices_scores = choices_scores / torch.pow(
                    (choices_ids != self.tokenizer.pad_token_id).sum(dim=-1), self.config.length_norm
                )