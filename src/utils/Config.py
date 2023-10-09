
import json
import os
import ast


class Config(object):
    def __init__(self, filenames=None, kwargs=None):
        # Experiment configs
        self.exp_dir = None
        self.exp_name = None
        self.allow_skip_exp = True
        self.seed = 42

        # Model Configs
        self.model = "EncDec"
        self.max_seq_len = 256
        self.origin_model = "bigscience/T0_3B"
        self.load_weight = ""

        # Dataset Configs
        self.dataset = "sst2"
        self.few_shot = True
        self.num_shot = 16
        self.few_shot_random_seed = 100
        self.train_template_idx = -1
        self.eval_template_idx = -1
        self.batch_size = 8
        self.eval_batch_size = 16
        self.num_workers = 8
        self.change_hswag_templates = False
        self.raft_cross_validation = True
        self.raft_validation_start = 0
        self.raft_labels_in_input_string = "comma"
        self.cleaned_answer_choices_b77 = False

        # something for co-training
        self.local_path = None

        # Compute backend configs
        self.compute_precision = "bf16"
        self.compute_strategy = "none"

        # Trainer configs
        self.num_steps = 300 # used for T0 fine-tuning
        self.num_epochs = 40 # used for label model fine-tuning
        self.eval_epoch_interval = 10_000
        self.eval_before_training = True
        self.save_model = True
        self.save_step_interval = 20_000
        self.mc_loss = 0
        self.unlikely_loss = 0
        self.length_norm = 0
        self.grad_accum_factor = 1
        self.split_option_at_inference = False  # Whether to split the answer choices during eval to lower memory usage for datasets with lots of answer choices

        # Optimization configs
        self.optimizer = "adafactor"
        self.lr = 3e-4
        self.trainable_param_names = ".*"
        self.scheduler = "linear_decay_with_warmup"
        self.warmup_ratio = 0.06
        self.weight_decay = 0
        self.scale_parameter = True
        self.grad_clip_norm = 1

        # PEFT method configs
        self.model_modifier = ""
        # Prompt Tuning configs
        self.prompt_tuning_num_prefix_emb = 100
        self.prompt_tuning_encoder = True
        self.prompt_tuning_decoder = True
        self.prompt_tuning_init_with_pad = False
        # LoRA configs
        self.lora_rank = 4
        self.lora_scaling_rank = 0
        self.lora_init_scale = 0.01
        self.lora_modules = "none"
        self.lora_layers = "none"
        # BitFit configs
        self.bitfit_modules = ".*"
        self.bitfit_layers = "q|k|v|o|wi_[01]|w_o"
        # Adapter configs
        self.adapter_type = "normal"
        self.adapter_non_linearity = "relu"
        self.adapter_reduction_factor = 4
        self.normal_adapter_residual = True
        self.lowrank_adapter_w_init = "glorot-uniform"
        self.lowrank_adapter_rank = 1
        self.compacter_hypercomplex_division = 8