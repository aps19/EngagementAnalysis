# utils/seed.py

import torch
import random
import numpy as np

def set_seed(seed: int):
    """
    Sets the seed for generating random numbers.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
