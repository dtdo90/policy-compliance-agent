"""Runtime helpers for CPU-bound model execution."""

from __future__ import annotations

import os


def configure_cpu_runtime(default_threads: int = 4) -> int:
    cpu_threads = int(os.environ.get("CPU_THREADS", str(default_threads)))
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        import torch

        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    return cpu_threads
