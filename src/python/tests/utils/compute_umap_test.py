"""Sanity-check tests for compute_umap.py after lazy migration.

compute_umap.py hardcodes the HPC input directory
(``/lustre06/project/.../epiclass/input``) and the chromsize path, so a
true end-to-end test from CI is not viable without that filesystem. We
keep an import smoke test here and skip the full run with a clear reason
— if the script is ever refactored to accept paths via the CLI, lift
the skip and run it against saccer3 fixtures.
"""
# pylint: disable=unused-import, import-outside-toplevel
import pytest


def test_compute_umap_imports():
    """Smoke test: module imports cleanly after the lazy migration."""
    import epiclass.utils.embedding.compute_umap


@pytest.mark.skip(
    reason="compute_umap.main() hardcodes /lustre06/project HPC paths; "
    "needs refactor before it can run against the saccer3 fixture."
)
def test_compute_umap_runs():
    """Placeholder for an end-to-end UMAP integration test."""
