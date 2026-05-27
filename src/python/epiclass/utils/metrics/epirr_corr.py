"""Create a correlation matrix based on the correlation of the signals of all datasets for a given epiRR (concatenated signals)"""
import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Collection, Dict, FrozenSet, List, Tuple

import numpy as np
import pandas as pd

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.argparseutils.directorychecker import DirectoryChecker
from epiclass.core.data_source import EpiDataSource
from epiclass.core.epiatlas_constants import TRACKS_MAPPING
from epiclass.core.lazy.lazy_hdf5_loader import LazyHdf5Loader
from epiclass.core.metadata import Metadata
from epiclass.utils.metadata_utils import EPIATLAS_ASSAYS as epiatlas_assays


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = ArgumentParser()

    # fmt: off
    arg_parser.add_argument(
        "hdf5", type=Path, help="A file with hdf5 filenames. Use absolute path!",
    )
    arg_parser.add_argument(
        "chromsize", type=Path, help="A file with chrom sizes.",
    )
    arg_parser.add_argument(
        "metadata", type=Path, help="A metadata JSON file.",
    )
    arg_parser.add_argument(
        "logdir", type=DirectoryChecker(), help="Directory for the output logs.",
    )
    # fmt: on
    return arg_parser.parse_args()


def print_unique_target_sets_distribution(epirr_dict: Dict[str, Dict]):
    """
    Prints the distribution of unique target/assay sets for all epiRRs.

    This function takes a dictionary containing EpiRR and their dsets and calculates the
    distribution of unique target sets (assays) for each EpiRR. It then prints the
    occurrence count, cumulative sum, and the sorted assays for each unique target set.

    Args:
        epirr_dict (Dict[str, Dict]): A dictionary where the keys are EpiRR IDs and the
            values are experiment metadata datasets.

    Example:
        epirr_dict = {
            "EPIRR00001": [{md5sum:val1, assay:val2, ...}, {md5sum:val3, assay:val4, ...}],
            "EPIRR00002": [dset1, dset2, dset3]
        }
        print_unique_target_sets_distribution(epirr_dict)
    """
    epirr_assays_dict = {
        epirr: frozenset([dset["assay_epiclass"] for dset in dset_list])
        for epirr, dset_list in epirr_dict.items()
    }
    occurences = Counter(assays for assays in epirr_assays_dict.values())

    cumsum = np.cumsum([n for _, n in occurences.most_common()])
    for i, (assays, n) in enumerate(occurences.most_common()):
        print(n, cumsum[i], sorted(assays))

    print(len(epirr_assays_dict))


def load_signals(
    datasource: EpiDataSource,
    signal_ids: List[str],
    mmap_dir: Path | None = None,
) -> Dict[str, np.ndarray]:
    """
    Load signals from HDF5 files using the provided EpiDataSource and
    a list of signal IDs which identify the files.

    Returns:
        dict: A dictionary containing the loaded signals, where the keys are the signal IDs
        and the values are the corresponding signals.
    """
    hdf5_loader = LazyHdf5Loader(
        datasource.chromsize_file, normalization=True, mmap_dir=mmap_dir
    )
    hdf5_loader.register_hdf5s(
        datasource.hdf5_file, signal_ids=signal_ids, strict=True, verbose=True
    )
    hdf5_loader.preload_all()
    return {sid: hdf5_loader.load_signal(sid) for sid in hdf5_loader.file_paths}


def define_accepted_tracks() -> FrozenSet[str]:
    """Create and return set of tracks accepted for this program."""
    accepted_tracks = set((list(TRACKS_MAPPING.keys()) + ["pval"]))
    accepted_tracks.remove("raw")
    accepted_tracks = frozenset(accepted_tracks)
    return accepted_tracks


class EpirrSignals:
    """Class to create epiRR signals based on the signals of all datasets for a given epiRR (concatenated signals)"""

    def __init__(
        self,
        metadata: Metadata,
        hdf5_signals: Dict[str, np.ndarray],
        accepted_targets: Collection,
    ):
        self.target_mapping = self.create_target_mapping(sorted(accepted_targets))
        self._epirr_signals, self._epirr_ids = self.create_epirr_signals(
            epirr_dict=self.group_by_epirr(metadata),
            signals_dict=hdf5_signals,
            targets_id=self.target_mapping,
        )
        """
        Initialize the EpirrSignals class.

        Args:
            metadata (Metadata): A Metadata instance containing information about the experiments.
            hdf5_signals (Dict[str, np.ndarray]): A dict mapping ids to signal arrays.
            accepted_targets (Collection): A collection of accepted targets/assays ("assay_epiclass")
        """

    @property
    def epirr_signals(self):
        """Returns the epiRR signals."""
        return self._epirr_signals

    @property
    def epirr_signal_ids(self):
        """Signal IDs used to create the epiRR signals."""
        return self._epirr_ids

    @staticmethod
    def group_by_epirr(metadata: Metadata) -> Dict[str, List]:
        """Return all datasets grouped by epiRR as new dictionnary"""
        epirr_dict = defaultdict(list)
        for dset in metadata.datasets:
            epirr = dset["epirr_id"]
            epirr_dict[epirr].append(dset)
        return epirr_dict

    @staticmethod
    def create_target_mapping(targets: List[str]) -> Dict[str, int]:
        """
        Create a target mapping for sections of the epiRR arrays (where to place signals).

        The targets list (mostly) defines the set and order of targets for the Epirr signal.

        Some pairs occupy the same zone:
         - WGBS-PBAT and WGBS-standard, standard is chosen over PBAT.
         - RNA-seq and mRNA-seq, RNA is chosen over mRNA.

        The number of expected zones/assays to represent an epiRR is hardcoded.

        Args:
            targets (List[str]): A list of target/assays strings.

        Returns:
            Dict[str, int]: A dictionary mapping target strings to integer indices.

        Raises:
            ValueError: If there are duplicate target values in the input list, or if the
                        target count does not match the expected number of Epirr array zones,
                        or if the target mapping does not match the expected integer range.
        """
        if len(targets) != len(set(targets)):
            raise ValueError(f"Incoherent target/assay values: {targets}")

        nb_zones = 9
        target_mapping: Dict[str, int] = {}
        for target in targets:
            if re.match(r"^\w?rna", target):
                target_mapping[target] = nb_zones - 2
                targets.remove(target)
            elif re.match(r"^wgbs", target):
                target_mapping[target] = nb_zones - 1
                targets.remove(target)

        if len(targets) != nb_zones:
            raise ValueError(
                f"{nb_zones} epirr array zones expected, {len(targets)} targets: {targets}"
            )

        for i, target in enumerate(targets):
            target_mapping[target] = i

        if set(target_mapping.values()) != set(range(0, nb_zones)):
            raise ValueError(
                f"Target mapping does not match expected range {set(target_mapping.values())} VS {set(range(0, nb_zones))}"
            )

        return target_mapping

    @staticmethod
    def create_epirr_signals(
        epirr_dict: Dict[str, List],
        signals_dict: Dict[str, np.ndarray],
        targets_id: Dict[str, int],
    ) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """
        Create a dictionary of the new epiRR signals representation.

        Args:
            epirr_dict (Dict[str, List]): A dict mapping EpiRR IDs to their list of datasets.
            signals_dict (Dict[str, np.ndarray]): A dict mapping ids to signals.
            targets_id (Dict[str, int]): A dictionary mapping target strings to integer indices.

        Returns:
            Dict[str, np.ndarray]: A dictionary mapping EpiRR IDs to signal arrays.
            List[str]: A list of all ids used in the creation of the different arrays.
        """
        nb_bins = list(signals_dict.values())[0].shape[0]
        array_length = len(targets_id) * nb_bins

        epirr_signals = {}
        used_signals = []
        for epirr, dset_list in epirr_dict.items():
            epirr_signal = np.empty(array_length)
            epirr_signal[:] = np.nan
            for dset in dset_list:
                target_id = targets_id[dset["assay_epiclass"]]
                sid = dset["md5sum"]
                used_signals.append((epirr, sid))

                start_index = target_id * nb_bins
                end_index = start_index + nb_bins
                epirr_signal[start_index:end_index] = signals_dict[sid]

            epirr_signals[epirr] = epirr_signal

        return epirr_signals, used_signals


def main():
    """main"""
    cli = parse_arguments()

    my_datasource = EpiDataSource(cli.hdf5, cli.chromsize, cli.metadata)
    my_metadata = Metadata(cli.metadata)
    log = cli.logdir

    accepted_tracks = define_accepted_tracks()

    my_metadata.select_category_subsets("track_type", accepted_tracks)
    my_metadata.select_category_subsets("assay_epiclass", epiatlas_assays)

    hdf5_signals = load_signals(
        my_datasource, list(my_metadata.signal_ids), mmap_dir=Path(log) / "mmap_cache"
    )
    my_metadata.select_category_subsets(
        "md5sum", list(hdf5_signals.keys())
    )  # TEMP FOR TEST

    epirr_signals_maker = EpirrSignals(my_metadata, hdf5_signals, epiatlas_assays)

    df = pd.DataFrame.from_dict(epirr_signals_maker.epirr_signals, orient="columns")

    correlated_signals = df.corr("pearson")

    correlated_signals.to_csv(log / "epirr_correlated_signals.csv")

    pd.DataFrame(epirr_signals_maker.epirr_signal_ids).to_csv(
        log / "epirr_correlated_signals.md5", index=False, header=False
    )


if __name__ == "__main__":
    main()
