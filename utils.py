"""Misc utilities: logging, seeding, I/O."""
import json
import os
import random
import time
from datetime import datetime

import numpy as np


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_metrics(metrics, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


class Timer:
    def __init__(self, name="block"):
        self.name = name
    def __enter__(self):
        self.t0 = time.time(); return self
    def __exit__(self, *a):
        log(f"{self.name} took {time.time()-self.t0:.1f}s")
