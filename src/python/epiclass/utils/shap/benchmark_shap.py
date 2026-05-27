"""Benchmarking SHAP value computation functions."""
from pathlib import Path

from epiclass.core.data_source import EpiDataSource
from epiclass.core.lazy.lazy_data_classes import LazyKnownData
from epiclass.core.lazy.lazy_epidata import DataSetFactory
from epiclass.core.metadata import Metadata
from epiclass.core.shap_values import NN_SHAP_Handler
from epiclass.utils.time import time_now


def benchmark(metadata: Metadata, datasource: EpiDataSource, model):
    """
    Benchmark the time taken for computing SHAP values based on the size of the background dataset.

    Args:
        metadata (Metadata): A Metadata object containing dataset metadata.
        datasource (EpiDataSource): An EpiDataSource object for accessing the data.
        model: The model used for computing SHAP values.
    """
    full_data = DataSetFactory.from_epidata(
        datasource=datasource,
        label_category="assay",
        metadata=metadata,
        min_class_size=1,
        test_ratio=0,
        validation_ratio=0,
        oversample=False,
    )

    for n in [250]:
        train_data = full_data.train.subsample(list(range(n)))
        shap_computer = NN_SHAP_Handler(model=model, logdir="")

        eval_size = 25
        evaluation_data = full_data.train.subsample(list(range(n, n + eval_size)))
        t_a = time_now()
        shap_computer.compute_shaps(
            background_dset=train_data,
            evaluation_dset=evaluation_data,
            save=False,
        )
        print(f"Time taken with n={n}: {time_now() - t_a}")


def test_background_effect(
    my_metadata: Metadata, my_datasource: EpiDataSource, my_model, logdir: Path
):
    """
    Test the effect of different background datasets on SHAP value computation.

    Args:
        my_metadata (Metadata): A Metadata object containing dataset metadata.
        my_datasource (EpiDataSource): An EpiDataSource object for accessing the data.
        my_model: The model used for computing SHAP values.
        logdir (Path): The path to the output log directory.
    """
    # --- Prefilter metadata ---
    my_metadata.display_labels("assay")
    my_metadata.select_category_subsets("track_type", ["pval", "Unique_plusRaw"])

    assay_list = ["h3k9me3", "h3k36me3", "rna_seq"]
    my_metadata.select_category_subsets("assay", assay_list)

    ids_per_class = my_metadata.ids_per_class("assay")
    background_1_signal_ids = ids_per_class["h3k9me3"][0:10]
    background_2_signal_ids = ids_per_class["rna_seq"][0:10]

    evaluation_signal_ids = (
        ids_per_class["h3k9me3"][10:20]
        + ids_per_class["rna_seq"][10:20]
        + ids_per_class["h3k36me3"][0:10]
    )
    all_signal_ids = set(
        background_1_signal_ids + background_2_signal_ids + evaluation_signal_ids
    )

    for sid in list(my_metadata.signal_ids):
        if sid not in all_signal_ids:
            del my_metadata[sid]

    full_data = DataSetFactory.from_epidata(
        datasource=my_datasource,
        label_category="assay",
        metadata=my_metadata,
        min_class_size=1,
        test_ratio=0,
        validation_ratio=0,
        oversample=False,
    )

    background_1_idxs = [
        i
        for i, signal_id in enumerate(full_data.train.ids)
        if signal_id in set(background_1_signal_ids)
    ]
    background_2_idxs = [
        i
        for i, signal_id in enumerate(full_data.train.ids)
        if signal_id in set(background_2_signal_ids)
    ]
    evaluation_idxs = list(
        set(range(full_data.train.num_examples))
        - set(background_1_idxs + background_2_idxs)
    )

    assert isinstance(full_data.train, LazyKnownData)
    background_1_data = full_data.train.subsample(background_1_idxs)
    background_2_data = full_data.train.subsample(background_2_idxs)

    evaluation_data = full_data.train.subsample(evaluation_idxs)

    for background_data in [background_1_data, background_2_data]:
        shap_computer = NN_SHAP_Handler(model=my_model, logdir=logdir)
        shap_computer.compute_shaps(
            background_dset=background_data,
            evaluation_dset=evaluation_data,
            save=True,
            name="background_effect_test",
        )
