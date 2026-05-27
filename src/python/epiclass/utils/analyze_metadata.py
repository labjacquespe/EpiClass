"""Analyze metadata content."""
# pylint: disable=import-error,import-outside-toplevel
import copy
from pathlib import Path

import pandas as pd

from epiclass.core.metadata import Metadata
from epiclass.utils import modify_metadata
from epiclass.utils.metadata_utils import EPIATLAS_CATS
from epiclass.utils.postprocessing.augment_predict_file import add_coherence

# from epiclass.utils.preconditions import check_epitatlas_uuid_premise


def two_step_long_analysis(my_metadata: Metadata, category1: str, category2: str):
    """Analyze impact of two-step filtering on metadata content with verbose prints."""
    print(
        f"First we remove small classes from {category1} and only keep datasets which have a {category2}."
    )
    my_metadata.remove_small_classes(10, category1)
    my_metadata.remove_missing_labels(category2)
    my_metadata.display_labels(category1)

    print(f"Then we examine the content of {category2} for each {category1}")
    nb_classes_kept = 0
    labels = my_metadata.label_counter(category1).keys()
    for label in sorted(labels):
        print(f"--{label}--")
        temp_metadata = copy.deepcopy(my_metadata)
        temp_metadata.select_category_subsets(label, [category1])
        temp_metadata.remove_small_classes(10, category2)
        if len(temp_metadata.label_counter(category2).most_common()) > 1:
            nb_classes_kept += 1
        temp_metadata.display_labels(category2)

    print(
        f"Number of classes with >1 cell_type label, from {category1}: {nb_classes_kept}/{len(labels)}"
    )


def two_step_table_analysis(my_metadata: Metadata, category1: str, category2: str):
    """Analyze impact of two-step filtering on metadata content. Produces a table.."""
    my_metadata.remove_small_classes(10, category1)
    my_metadata.remove_small_classes(10, category2)
    my_metadata.remove_missing_labels(category2)

    counter_cat1 = my_metadata.label_counter(category1)

    info = {}
    for label in sorted(counter_cat1.keys()):
        info[label] = {}
        info[label]["total_examples_before_step2_filter"] = counter_cat1[label]

        temp_metadata = copy.deepcopy(my_metadata)
        temp_metadata.select_category_subsets(label, [category1])

        info[label]["total_classes_before_step2_filter"] = len(
            temp_metadata.label_counter(category2)
        )

        temp_metadata.remove_small_classes(10, category2)

        counter_cat2 = temp_metadata.label_counter(category2)
        info[label]["total_examples_after_step2_filter"] = sum(counter_cat2.values())
        info[label]["total_classes_after_step2_filter"] = len(counter_cat2)

    line = "\t{total_classes_before_step2_filter}\t{total_classes_after_step2_filter}\t{total_examples_before_step2_filter}\t{total_examples_after_step2_filter}\t"
    print(
        "class_step1\t#labels_before\t#labels_after\t#dsets_before\t#dsets_after\tratio #dsets/#labels"
    )
    for label, infos in info.items():
        try:
            ratio = (
                float(infos["total_examples_after_step2_filter"])
                / infos["total_classes_after_step2_filter"]
            )
        except ZeroDivisionError:
            ratio = "--"
        print(label + line.format(**infos) + str(ratio))


def compute_coherence_on_all(meta: Metadata):
    """Compute coherence on all metadata categories."""
    for dset in list(meta.datasets):
        if dset["track_type"] in set({"pval", "fc"}):
            del meta[dset["md5sum"]]

    df = pd.DataFrame(list(meta.datasets))  # type: ignore
    df = df.replace(r"^\s*$", "--", regex=True)

    for category in list(df.columns):
        if category not in EPIATLAS_CATS | set(
            [
                "md5sum",
                "EpiRR",
                "epirr_id",
                "track_type",
                "assay",
            ]
        ):
            add_coherence(df, category)

    df.drop(df.filter(regex="Coherence count").columns, axis=1, inplace=True)  # type: ignore

    df.to_csv("test.csv", index=False)


def create_json_from_signal_id_list(signal_id_list: Path, metadata: Metadata):
    """Save json with metadata from selected signals."""
    metadata = copy.deepcopy(metadata)
    with open(signal_id_list, "r", encoding="utf8") as f:
        wanted_ids = set(line.strip() for line in f.readlines())

    for signal_id in list(metadata.signal_ids):
        if signal_id not in wanted_ids:
            del metadata[signal_id]

    new_path = signal_id_list.parent / f"{signal_id_list.stem}-metadata.json"
    metadata.save(new_path)


def test_generic_classifiers(my_metadata: Metadata):
    """Multiple tests to understand how much correct answers you could get
    from statistical derivations, without making any assumptions about the actual data."""
    import numpy as np
    import scipy
    import sklearn
    from statsmodels.multivariate.manova import MANOVA

    modify_metadata.merge_pair_end_info(my_metadata)
    my_metadata.select_category_subsets("paired_end_mode", ["single_end", "paired_end"])

    # -- test 1 with simple classifiers --
    df = pd.DataFrame(my_metadata.datasets)
    y = df["paired_end_mode"]
    y = sklearn.preprocessing.LabelEncoder().fit_transform(y)
    classifier = sklearn.linear_model.LogisticRegression(penalty="none")
    classifier = sklearn.ensemble.RandomForestClassifier(
        class_weight="balanced_subsample"
    )
    correlations = []
    for category in my_metadata.get_categories():
        X = df[category]
        X = sklearn.preprocessing.LabelEncoder().fit_transform(X).reshape(-1, 1)
        acc = []
        classifier = classifier.fit(X, y)
        for _ in range(10):
            X, y = sklearn.utils.shuffle(X, y)
            y_pred = classifier.predict(X)
            acc_1 = sklearn.metrics.accuracy_score(y, y_pred)
            adj_acc = sklearn.metrics.balanced_accuracy_score(y, y_pred, adjusted=True)
            acc.append((acc_1, adj_acc))
        acc = np.array(acc)
        mean_acc = acc.mean(axis=0)
        std = acc.std(axis=0)[0]
        correlations.append((category, str(mean_acc[0]), str(mean_acc[1]), str(std)))

    for metrics in correlations:
        print(" ".join(metrics))

    # -- test 2: Evaluate chi2 --
    import statsmodels.api as sm

    pivot = pd.crosstab(
        df["sample_ontology_term_high_order_unique"], df["paired_end_mode"]
    )
    print(pivot)
    chi2_stat, _, _, expected = scipy.stats.chi2_contingency(pivot)
    print(chi2_stat, expected)

    table = sm.stats.Table.from_data(
        df[["sample_ontology_term_high_order_unique", "paired_end_mode"]]
    )
    print(table.table_orig, "\n")
    print(table.fittedvalues, "\n")
    print(table.resid_pearson, "\n")
    print(table.chi2_contribs, "\n")

    # -- test 3: MANOVA --
    # encode labels
    df = df.apply(lambda x: pd.factorize(x)[0])
    maov = MANOVA.from_formula(
        "sample_ontology_term_high_order_unique ~ paired_end_mode", data=df
    )
    print(maov.mv_test())


def main():
    """Main function."""
    base = Path("/home/local/USHERBROOKE/rabj2301/Projects/epilap/input/metadata")
    path = base / "hg38_2023_epiatlas_dfreeze_plus_encode_noncore_formatted_JR.json"
    my_metadata = Metadata(path)

    # signal_id_list = "/home/local/USHERBROOKE/rabj2301/Projects/sources/epiclass/src/python/tests/fixtures/test-epilap-empty-biotype-n40.md5"
    # create_json_from_signal_id_list(Path(signal_id_list), my_metadata)

    # my_metadata.display_labels("track_type")
    # my_metadata.display_labels("harmonized_sample_ontology_intermediate")
    my_metadata.display_labels("assay_epiclass")
    # check_epitatlas_uuid_premise(my_metadata)

    # for label in my_metadata.get_categories():
    # print(label)


if __name__ == "__main__":
    main()
