"""Annotate Comet experiments whose name says "oversampling" but that were not oversampled.

Adds, without changing any existing value:
- the tag TAG;
- the "other" key OTHER_KEY, with a short explanation (OTHER_VALUE).

Each experiment is checked first, and skipped unless:
- its name contains "oversampling";
- its logged parameter `hparams|oversampling` is false.

Dry run by default: prints what would be done. Pass --apply to write to Comet.

Usage:
    python annotate_not_oversampled.py experiment_keys.txt           # dry run
    python annotate_not_oversampled.py experiment_keys.txt --apply   # write

experiment_keys.txt: one experiment key per line, empty lines and lines starting
with "#" are ignored.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import List

from comet_ml.api import API

TAG = "incoherent_name-vs-oversampling"
OTHER_KEY = "incoherence_note"
OTHER_VALUE = (
    f"{date.today().isoformat()}: the name and logdir say 'oversampling', but this run "
    "was not oversampled (hparams|oversampling=false, train size + validation size = "
    "total nb of files). Name and logdir left unchanged."
)


def read_experiment_keys(path: Path) -> List[str]:
    """Return the experiment keys listed in a text file, one per line."""
    lines = [line.strip() for line in path.read_text(encoding="utf8").splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def oversampling_parameter(parameters_summary: List[dict]) -> str | None:
    """Return the logged value of the oversampling hparam, or None if it was not logged."""
    for item in parameters_summary:
        if item["name"] in ("hparams|oversampling", "hparams/oversampling"):
            return str(item["valueCurrent"]).lower()
    return None


def main() -> None:
    """Check each listed experiment, then add the tag and the other if --apply is given."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "keys_file", type=Path, help="Text file, one experiment key per line."
    )
    parser.add_argument(
        "--apply", action="store_true", help="Write to Comet (default: dry run)."
    )
    args = parser.parse_args()

    api = API(cache=False)
    experiment_keys = read_experiment_keys(args.keys_file)
    print(f"{len(experiment_keys)} experiment keys read. Apply: {args.apply}\n")

    nb_annotated = 0
    for key in experiment_keys:
        experiment = api.get_experiment_by_key(key)
        if experiment is None:
            print(f"SKIP {key}: experiment not found.")
            continue

        name = experiment.name
        oversampling = oversampling_parameter(experiment.get_parameters_summary())

        if "oversampling" not in name:
            print(f"SKIP {key}: name does not mention oversampling ({name}).")
            continue
        if oversampling != "false":
            print(
                f"SKIP {key}: hparams|oversampling is {oversampling}, not false ({name})."
            )
            continue

        if args.apply:
            experiment.add_tags([TAG])
            experiment.log_other(OTHER_KEY, OTHER_VALUE)
            print(f"DONE {key}: {name}")
        else:
            print(f"WOULD ANNOTATE {key}: {name}")
        nb_annotated += 1

    verb = "Annotated" if args.apply else "Would annotate"
    print(f"\n{verb} {nb_annotated} of {len(experiment_keys)} experiments.")


if __name__ == "__main__":
    main()
