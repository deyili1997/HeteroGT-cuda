import os
import random
import numpy as np
import torch


def set_random_seed(seed: int):
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.
    This function ensures deterministic behavior for experiments,
    which is essential for debugging and consistent results.

    Parameters:
    -----------
    seed : int
        The seed value to use for all relevant random number generators.
    """

    # Set the Python built-in random module's seed
    random.seed(seed)

    # Set the NumPy random seed
    np.random.seed(seed)

    # Set the PyTorch random seed for CPU computations
    torch.manual_seed(seed)

    # If running on a machine with Apple Silicon (M1/M2/M3),
    # PyTorch uses Metal Performance Shaders (MPS) backend instead of CUDA.
    # CUDA seed calls are skipped in this case.
    if torch.backends.mps.is_available():
        print("[INFO] Detected Apple MPS backend (Mac with M1/M2/M3). Skipping CUDA seeds.")
    
    # If CUDA (NVIDIA GPU) is available — for external GPU cases on Mac/Linux/Windows
    elif torch.cuda.is_available():
        # Set CUDA random seed for the current device
        torch.cuda.manual_seed(seed)
        # Set CUDA random seed for all devices (multi-GPU training)
        torch.cuda.manual_seed_all(seed)

    # For consistent results from cuDNN (used internally by PyTorch for some ops)
    # Set deterministic mode to True to ensure reproducibility
    torch.backends.cudnn.deterministic = True

    # Disable cuDNN benchmarking — it selects the fastest algorithm at runtime
    # which can introduce randomness in timing or behavior
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed to ensure that any hash-based ops (like dict order)
    # are consistent across runs
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Optional (for logging/debugging)
    print(f"[INFO] Random seed set to {seed}")