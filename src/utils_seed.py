# utils_seed.py
import os
import random
import hashlib
import numpy as np
import torch

def make_seed(base_seed: int, condition: str="", run_id: int=1) -> int:
    """
    Derive a stable 32-bit seed from (base_seed, condition string, run_id).
    This is deterministic across machines and Slurm jobs.
    """
    s = f"{base_seed}|{condition}|{run_id}".encode("utf-8")
    h = hashlib.sha256(s).hexdigest()
    return int(h[:8], 16)  # 32-bit

def seed_everything(seed: int, deterministic: bool = False) -> None:
    """
    Apply seeds for Python, NumPy, PyTorch (CPU/CUDA) and optionally enforce
    deterministic algorithms (slower; may raise errors for non-deterministic ops).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # These env vars are safest if set before CUDA context is created.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

def seed_worker(worker_id: int) -> None:
    """
    Ensures each DataLoader worker has a deterministic numpy/random seed.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_loader(dataset, batch_size, shuffle, seed, num_workers=0):
    g = torch.Generator()
    g.manual_seed(seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
    )
