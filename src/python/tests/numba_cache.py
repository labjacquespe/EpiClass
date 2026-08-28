"""Force numba's on-disk kernel cache for the JIT-heavy embedding tests.

Why this exists
---------------
umap-learn 0.5.12 declares ``cache=True`` on none of its 117 ``@numba.njit``
functions, and pynndescent 0.6.0 on only 20 of its 227. Every fresh process
therefore recompiles ~100 kernels from scratch. That compilation -- not the
numeric work, which is ~2s on the 100-sample saccer3 subset -- is what makes
``test_compute_umap_single_sample`` the longest test in the suite by a wide
margin.

``cache=True`` is only a decorator kwarg, and all 344 of those declarations
use the attribute form ``@numba.njit`` on a plain ``import numba``. So
wrapping that single attribute *before umap is imported* reaches all of them
without patching the installed packages.

Measured on the full suite, 4 workers, median of 2 rounds:
111.0s -> 98.2s wall once the cache is warm. The run that populates the cache
pays ~11s extra. Per-process, the two UMAP tests together go 66.9s -> 42.3s.

Three pynndescent kernels (``nn_descent``, ``nn_descent_internal``,
``search_closure``) build closures over dynamic globals, and numba refuses to
cache those. They still recompile in every process; the numbers above are what
remains after that irreducible part.

Concurrency
-----------
Numba writes cache files atomically (temp file + ``os.replace``), so a torn
read is not possible. The race that *is* possible is a lost update: saving an
overload is a read-modify-write of the index file, so two processes saving at
the same time can have one silently drop the other's entry. The cost is a
recompile, not a wrong result -- but with xdist there is no reason to accept
even that, so ``Cache.save_overload`` is serialized here under an exclusive
flock shared by every worker.

In practice one worker runs both UMAP tests (they are kept adjacent in
collection order on purpose -- see ``conftest.LONG_FIRST_NODEIDS``), so today
the contention window is narrow. The lock is here for the general case: any
other cached numba code, in any two workers, would otherwise race.

Switches
--------
``EPICLASS_NO_NUMBA_CACHE=1``  disable entirely. A cold cache is a net loss
(~11s), so CI without a persisted cache directory wants this set.
``EPICLASS_NUMBA_CACHE=<dir>`` relocate the cache (defaults to
``tests/.numba_cache``). On HPC, point it somewhere node-local.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, Callable, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

DEFAULT_CACHE_DIR = Path(__file__).parent / ".numba_cache"
LOCK_NAME = ".index.lock"

_installed = False


def cache_dir() -> Path:
    """Directory numba should keep compiled kernels in."""
    override = os.environ.get("EPICLASS_NUMBA_CACHE")
    return Path(override) if override else DEFAULT_CACHE_DIR


def _force_cache(orig: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a numba jit decorator so every declaration opts into the cache."""

    def wrapper(*args, **kwargs):
        # Bare form: @numba.njit applied straight to the function.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return orig(cache=True)(args[0])
        # Called form: @numba.njit(...), with or without an explicit signature.
        # Never override an explicit cache= from the library itself.
        kwargs.setdefault("cache", True)
        return orig(*args, **kwargs)

    return wrapper


@contextlib.contextmanager
def _index_lock(lock_path: Path) -> Iterator[None]:
    """Hold an exclusive cross-process lock, or nothing if flock is missing."""
    if fcntl is None:  # pragma: no cover - Windows
        yield
        return
    with open(lock_path, "w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _serialize_writes(orig: Callable[..., Any], lock_path: Path) -> Callable[..., Any]:
    """Serialize save_overload's read-modify-write of the cache index."""

    def wrapper(self, sig, data):
        with _index_lock(lock_path):
            return orig(self, sig, data)

    return wrapper


def install() -> bool:
    """Patch numba to cache compiled kernels on disk. Return True if applied.

    Must run before umap/pynndescent are imported: the decorators execute at
    *their* import time, so a later patch would be too late. tests/conftest.py
    calls this from pytest_configure, which runs before collection imports any
    test module.
    """
    global _installed  # pylint: disable=global-statement
    if _installed or os.environ.get("EPICLASS_NO_NUMBA_CACHE"):
        return False

    target = cache_dir()
    target.mkdir(parents=True, exist_ok=True)

    # numba reads this when it is imported, so it must be set first.
    os.environ["NUMBA_CACHE_DIR"] = str(target)

    import numba  # pylint: disable=import-outside-toplevel
    import numba.core.caching  # pylint: disable=import-outside-toplevel

    numba.njit = _force_cache(numba.njit)
    numba.jit = _force_cache(numba.jit)
    numba.core.caching.Cache.save_overload = _serialize_writes(
        numba.core.caching.Cache.save_overload, target / LOCK_NAME
    )

    _installed = True
    return True
