# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import random

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task

import torch
# home cartesian
# pose:  tensor([[-0.1086,  0.0167,  0.4626],
#         [-0.1086,  0.0167,  0.4626],
#         [-0.1086,  0.0167,  0.4626],
#         [-0.1086,  0.0167,  0.4626],
#         [-0.1086,  0.0167,  0.4626],
#         [-0.1086,  0.0167,  0.4626],
#         [-0.1086,  0.0167,  0.4626],
#         [-0.1086,  0.0167,  0.4626]], device='cuda:0')
# ori:  tensor([[ 0.4696, -0.5536, -0.3459,  0.5944],
#         [ 0.4696, -0.5536, -0.3459,  0.5944],
#         [ 0.4696, -0.5536, -0.3459,  0.5944],
#         [ 0.4696, -0.5536, -0.3459,  0.5944],
#         [ 0.4696, -0.5536, -0.3459,  0.5944],
#         [ 0.4696, -0.5536, -0.3459,  0.5944],
#         [ 0.4696, -0.5536, -0.3459,  0.5944],
#         [ 0.4696, -0.5536, -0.3459,  0.5944]], device='cuda:0')

def train():
    task, env = parse_task(args, cfg, cfg_train, sim_params)
    home = torch.tensor([[-0.1086,  0.0167,  0.4626 , 0.4696, -0.5536, -0.3459,  0.5944],
                         [-0.1086,  0.0167,  0.4626 , 0.4696, -0.5536, -0.3459,  0.5944],
                         [-0.1086,  0.0167,  0.4626 , 0.4696, -0.5536, -0.3459,  0.5944],
                         [-0.1086,  0.0167,  0.4626 , 0.4696, -0.5536, -0.3459,  0.5944],
                         [-0.1086,  0.0167,  0.4626 , 0.4696, -0.5536, -0.3459,  0.5944],
                         [-0.1086,  0.0167,  0.4626 , 0.4696, -0.5536, -0.3459,  0.5944],
                         [-0.1086,  0.0167,  0.4626 , 0.4696, -0.5536, -0.3459,  0.5944],
                         [-0.1086,  0.0167,  0.4626 , 0.4696, -0.5536, -0.3459,  0.5944]] ,device=env.rl_device) 
    for i in range( 10000):
        obs, _, _, _, _ = env.step(home) # todo notice actions will be clipped
        #print(obs.shape)

    


if __name__ == '__main__':
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    train()
