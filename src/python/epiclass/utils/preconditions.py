"""Check for specific conditions before running the main program, .e.g git in clean state, metadata passes all test flags, etc.

Uses GitPython package.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

from git import Repo

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.core.metadata import Metadata


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = ArgumentParser()

    # fmt: off
    arg_parser.add_argument(
        "-m", "--metadata", type=Path, help="A metadata JSON file to analyze for errors.",
    )
    # fmt: on
    return arg_parser.parse_args()


def find_upwards(cwd: Path, filename: str) -> Path | None:
    """Python: search for a file in current directory and all it's parents
    https://stackoverflow.com/a/70859914/11472153"""
    if cwd == Path(cwd.root) or cwd == cwd.parent:
        return None

    fullpath = cwd / filename

    return fullpath if fullpath.exists() else find_upwards(cwd.parent, filename)


def git_repo_is_clean(verbose=True) -> bool:
    """Verifies that the git has no staged or tracked unstaged changes.

    Return True if condition verified.
    """
    is_clean = True
    dir_here = Path(__file__).resolve()
    git_dir = find_upwards(dir_here, ".git")
    if git_dir is None:
        raise OSError("Could not find git directory.")

    repo = Repo(git_dir)

    # Verify if staged changes exist
    staged = [item.a_path for item in repo.index.diff("HEAD")]
    if staged:
        is_clean = False
        if verbose:
            print(f"Staged files exist: {staged}")

    # Verify if tracked+unstaged changes exist
    unstaged = [item.a_path for item in repo.index.diff(None)]
    if unstaged:
        is_clean = False
        if verbose:
            print(f"Tracked+unstaged changes exist: {unstaged}")

    return is_clean


def check_epitatlas_uuid_premise(metadata: Metadata):
    """Check that there is only one file per track type, for a given uuid."""

    uuid_to_track_count = collections.defaultdict(collections.Counter)
    uuid_to_signal_ids = collections.defaultdict(list)
    for signal_id, dset in metadata.items:
        uuid = dset["uuid"]
        track_type = dset["track_type"]

        uuid_to_track_count[uuid].update([track_type])
        uuid_to_signal_ids[uuid].append((signal_id, track_type))

    is_okay = True
    bad_uuid = []
    for uuid, counter in uuid_to_track_count.items():
        for nb in counter.values():
            if nb != 1:
                print(uuid, counter)
                bad_uuid.append(uuid)
                break

    if bad_uuid:
        is_okay = False

    for uuid in bad_uuid:
        print(f"Problematic uuid: {uuid}")
        print(uuid_to_track_count[uuid])
        for signal_id, track_type in uuid_to_signal_ids[uuid]:
            print(signal_id, track_type)

    return is_okay


def main():
    """Exit with error if a precondition is not met."""

    cli = parse_arguments()
    print("\n")
    pass_preconditions = []
    if not git_repo_is_clean():
        print("Git directory not in a clean state.\n")
        pass_preconditions.append(False)

    metadata_path = cli.metadata
    if metadata_path is not None:
        metadata_path = metadata_path.resolve()

        metadata = Metadata(metadata_path)
        if not check_epitatlas_uuid_premise(metadata):
            pass_preconditions.append(False)
            print("Metadata did not pass uuid check (1 file per track type per uuid).\n")

    if not all(pass_preconditions):
        print("A precondition was not met, exiting with error.")
        sys.exit(1)


if __name__ == "__main__":
    main()
