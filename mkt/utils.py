import os
import random
from datetime import datetime

import numpy as np
import torch

import wandb

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def wandb_init_distributed(args, script_filename):
    wandb.init(
        project=args.project,
        entity=args.entity,
        name=f"{os.path.basename(script_filename)}_{args.local_rank}",
        group=f"{args.group}_{datetime.now().strftime('%Y-%m-%d %H:%M')}",
    )