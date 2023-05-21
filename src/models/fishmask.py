import os
import torch


def fishmask_plugin_on_init(pl_module):
    if pl_module.config.fishmask_mode == "apply":
   