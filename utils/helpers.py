import torch
import numpy as np
import random
import logging

logger = logging.getLogger('experiment')

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_run(all_args, rank):
    args = all_args.copy()
    args["seed"] = args["seed"][rank]
    return args

def memory_usage():
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 ** 2
    except ImportError:
        logger.warning("psutil not installed. Cannot report memory usage.")
        return 0