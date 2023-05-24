import os
import torch


def fishmask_plugin_on_init(pl_module):
    if pl_module.config.fishmask_mode == "apply":
        print(f"Load gradient mask from {pl_module.config.fishmask_path}")
        mask_dict = torch.load(pl_module.config.fishmask_path)
        for param_name, param in pl_module.model.named_parameters():
            param.stored_mask = mask_dict[param_name].to("cuda")


def fishmask_plug