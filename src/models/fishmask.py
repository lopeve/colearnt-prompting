import os
import torch


def fishmask_plugin_on_init(pl_module):
    if pl_module.config.fishmask_mode == "apply":
        print(f"Load gradient mask from {pl_module.config.fishmask_path}")
        mask_dict = torch.load(pl_module.config.fishmask_path)
        for param_name, param in pl_module.model.named_parameters():
            param.stored_mask = mask_dict[param_name].to("cuda")


def fishmask_plugin_on_optimizer_step(pl_module):
    if pl_module.config.fishmask_mode == "create":
        for name, param in pl_module.model.named_parameters():
            if not hasattr(param, "stored_grad"):
                param.stored_grad = torch.zeros_like(param.data)
            param.stored_grad += torch.square(param.grad) / pl_module.config.num_shot
            param.grad.zero_()
    elif pl_module.config.fishmask_mode == "apply":
        for name, param in pl_module.model.named_parameters():
            param.grad.data *= param.stored_mask
    else:
        raise ValueError(f"Invalid fishmask_mode {pl_module.config.fishmask_mode}")


def fishmask_plugin_on_end(pl_module):
    if pl_module.config.fishmask_mode == "create":
        sizes = {}
        tensors = []
        all_params_size = 0
        for param_name, param in pl_module.model.named_parameters():
            sizes[param_name] = param.size()
     