"""Guard against the BLIS < 1.1 CPU-GEMM segfault.

BLIS < 1.1 has an out-of-bounds read in its single-precision Haswell ``gemmsup`` kernels
that segfaults fp32 CPU matrix multiplies at realistic sizes (notably the FlexiBLAS-bundled
``bliscore/0.9.0`` on the Digital Research Alliance of Canada clusters). It bites CPU
*inference* (``predict.py`` / ``predict_CV.py``) and the forced-CPU validation prediction at
the end of a training run, so both the prediction and training entry points call
:func:`check_blas_backend` up front to fail fast with actionable instructions instead of
crashing deep into a run after the long data load.
"""
from __future__ import annotations

import ctypes
import os
import re
from typing import Optional

import torch


def _env_flag(name: str) -> bool:
    """Return True if environment variable ``name`` is set to a truthy value."""
    return os.getenv(name, "") not in ("", "0", "false", "False")


def _active_blis_version() -> Optional[str]:
    """Return the version of the BLIS library currently mapped into the process, else None.

    FlexiBLAS loads its BLAS backend lazily, so trigger a small matmul before calling this.
    Returns None when no BLIS library is mapped (OpenBLAS, MKL, netlib, GPU run, or a
    non-Linux platform without ``/proc``) or when the version cannot be determined.
    """
    try:
        with open("/proc/self/maps", encoding="utf8") as handle:
            libblis_paths = sorted(
                {line.split()[-1] for line in handle if "libblis" in line and "/" in line}
            )
    except OSError:
        return None
    for path in libblis_paths:
        try:  # ask the library itself — robust regardless of install path naming
            lib = ctypes.CDLL(path)
            lib.bli_info_get_version_str.restype = ctypes.c_char_p
            match = re.match(
                r"(\d+\.\d+(?:\.\d+)?)", lib.bli_info_get_version_str().decode()
            )
            if match:
                return match.group(1)
        except (OSError, AttributeError, ValueError, UnicodeDecodeError):
            pass
        match = re.search(
            r"/(?:bliscore|blis)/(\d+\.\d+(?:\.\d+)?)/", path
        )  # fall back to path
        if match:
            return match.group(1)
    return None


def check_blas_backend() -> None:
    """Refuse to run on a BLIS BLAS backend older than 1.1.

    BLIS < 1.1 has an out-of-bounds read in its Haswell ``gemmsup`` kernels that segfaults
    fp32 CPU matrix multiplies at realistic sizes (notably the FlexiBLAS-bundled
    ``bliscore/0.9.0`` on the Digital Research Alliance of Canada clusters). Detect the
    active backend up front and stop with actionable instructions, rather than crashing
    deep into a run after the long data load. Set ``EPICLASS_ALLOW_BAD_BLIS`` to bypass.

    Keys off the CPU BLAS library, not the compute device: a GPU training run still loads
    the CPU BLAS for the forced-CPU validation prediction at the end of training, so the
    check is relevant there too.
    """
    if _env_flag("EPICLASS_ALLOW_BAD_BLIS"):
        return
    # Force FlexiBLAS to dispatch a real sgemm so its lazy backend is loaded and shows up
    # in /proc/self/maps. Square/small-K, so it cannot hit the buggy skinny-sup path.
    _ = torch.ones((128, 128)) @ torch.ones((128, 128))
    version = _active_blis_version()
    if version is None:
        return
    if tuple(int(part) for part in version.split(".")) < (1, 1):
        raise RuntimeError(
            f"Active BLAS backend is BLIS {version}, which has a Haswell gemmsup "
            f"out-of-bounds read that segfaults CPU inference (fixed in BLIS 1.1). "
            f"Re-run with a safe backend: `export FLEXIBLAS=openblas` "
            f"(or set EPICLASS_ALLOW_BAD_BLIS=1 to bypass this check)."
        )
    print(f"BLAS backend: BLIS {version}")


__all__ = ["check_blas_backend"]
