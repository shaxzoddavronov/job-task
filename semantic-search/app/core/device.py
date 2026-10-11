"""Pick the compute device: CUDA when available (or forced), otherwise CPU."""

import logging

log = logging.getLogger(__name__)


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:  # torch is a runtime dependency; this only guards odd installs
        return False
    return bool(torch.cuda.is_available())


def resolve_device(setting: str) -> str:
    """'auto' → 'cuda' if a GPU is usable, else 'cpu'. 'cuda' without a GPU is an error."""
    setting = setting.lower()
    if setting == "auto":
        return "cuda" if cuda_available() else "cpu"
    if setting == "cuda" and not cuda_available():
        raise RuntimeError("DEVICE=cuda but no CUDA GPU is available (use DEVICE=auto or cpu)")
    if setting not in ("cuda", "cpu"):
        raise ValueError(f"DEVICE must be auto, cuda or cpu, not {setting!r}")
    return setting


def describe_device(device: str) -> dict:
    """Device facts for logs and /health (GPU name and memory when on CUDA)."""
    info: dict = {"device": device}
    if device == "cuda":
        import torch

        free, total = torch.cuda.mem_get_info()
        info |= {
            "gpu": torch.cuda.get_device_name(0),
            "vram_total_mb": total // 2**20,
            "vram_free_mb": free // 2**20,
        }
    return info


def configure_cpu_threads(threads: int) -> None:
    """Limit torch CPU threads (0 = leave torch's default)."""
    if threads > 0:
        import torch

        torch.set_num_threads(threads)
        log.info("torch CPU threads: %d", threads)
