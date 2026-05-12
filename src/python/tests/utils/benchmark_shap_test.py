"""Sanity-check integration test for benchmark_shap.py after lazy migration.

benchmark_shap has no main() — it exposes ``benchmark()`` and
``test_background_effect()`` helpers. ``benchmark()`` hardcodes n=250
background samples which doesn't fit our small test fixtures, so we
exercise the same migration surface (lazy DataSetFactory + LazyKnownData
isinstance + NN_SHAP_Handler) inline with subsample sizes that fit.
"""
from pathlib import Path

import pytest

from epiclass.core.lazy.lazy_data_classes import LazyKnownData
from epiclass.core.lazy.lazy_epidata import DataSetFactory
from epiclass.core.shap_values import NN_SHAP_Handler


@pytest.fixture(name="test_dir")
def fixture_test_dir(mk_logdir) -> Path:
    """Make temp logdir for tests."""
    return mk_logdir("benchmark_shap")


@pytest.mark.slow
def test_benchmark_shap_pipeline(
    test_epiatlas_data_handler, test_NN_model, test_dir: Path
):
    """End-to-end: lazy DataSetFactory.from_epidata → subsample → SHAP compute.

    Mirrors the steps in benchmark_shap.benchmark() with sizes that fit the
    40-sample test fixture (the real benchmark uses n=250).
    """
    datasource = test_epiatlas_data_handler.epiatlas_dataset.datasource
    metadata = test_epiatlas_data_handler.epiatlas_dataset.metadata

    full_data = DataSetFactory.from_epidata(
        datasource=datasource,
        label_category="biomaterial_type",  # matches the test fixture's category
        metadata=metadata,
        min_class_size=1,
        test_ratio=0,
        validation_ratio=0,
        oversample=False,
        mmap_dir=test_dir / "mmap_cache",
    )

    assert isinstance(full_data.train, LazyKnownData)

    n_bg, n_eval = 5, 2
    background = full_data.train.subsample(list(range(n_bg)))
    evaluation = full_data.train.subsample(list(range(n_bg, n_bg + n_eval)))

    handler = NN_SHAP_Handler(model=test_NN_model, logdir=test_dir)
    _, shap_values = handler.compute_shaps(
        background_dset=background,
        evaluation_dset=evaluation,
        save=False,
        num_workers=1,
    )
    assert shap_values.shape[0] == n_eval
