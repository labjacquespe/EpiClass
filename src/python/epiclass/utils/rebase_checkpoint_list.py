"""Interactive utility to repair ``best_checkpoint.list`` files after a move.

``best_checkpoint.list`` (written by ``MyTrainer.save_model_path``,
core/trainer.py) stores **absolute** checkpoint paths, one per line, in the
format ``"<ckpt_path> <iso_timestamp>"``. Model restoration reads the last line
and loads that ``.ckpt`` (see ``core/model_checkpoint.py``). Because the paths
are absolute, moving or copying a model directory breaks them: the suffix
(``EpiLaP/<hash>/checkpoints/epoch=...ckpt``) stays stable, only the leading
base directory changes.

The list file is always kept at a fixed position relative to its checkpoints,
so the directory holding ``best_checkpoint.list`` *is* the correct new base.
This script detects the stale old base automatically -- it finds the longest
trailing suffix of a stored path that actually exists under the list file's own
directory -- then swaps that old base for the new one on every matching line.

When the exact checkpoint of the last entry cannot be found at all (e.g. cleanups keeping only ``last.ckpt``),
``--fallback-ckpt`` locates the checkpoint directory under the new base and
*appends* a new line pointing at the surviving file (``last.ckpt`` by default),
so restoration -- which reads the last line -- picks it up.

Usage::

    python -m epiclass.utils.rebase_checkpoint_list LIST_FILE [LIST_FILE ...] \\
        [--dry-run] [--yes] [--no-backup] [--fallback-ckpt [NAME]]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

LIST_NAME = "best_checkpoint.list"


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument(
        "list_files",
        metavar="LIST_FILE",
        nargs="+",
        type=Path,
        help=f"One or more {LIST_NAME} files to repair.",
    )
    arg_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the proposed rewrite without writing anything.",
    )
    arg_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt (for scripting).",
    )
    arg_parser.add_argument(
        "--no-backup",
        action="store_true",
        help=f"Do not keep a '{LIST_NAME}.bak' copy of the original.",
    )
    arg_parser.add_argument(
        "--fallback-ckpt",
        nargs="?",
        const="last.ckpt",
        default=None,
        metavar="NAME",
        help=(
            "When the last entry's checkpoint cannot be located (e.g. it was "
            "deleted, leaving only last.ckpt), append a new line pointing at "
            "NAME found in the matching checkpoint directory. Defaults to "
            "'last.ckpt' when given with no value; disabled when omitted."
        ),
    )
    return arg_parser.parse_args()


def parse_line(line: str) -> Tuple[str, str]:
    """Split a list line into ``(ckpt_path, rest)``.

    Matches the parsing in ``core/model_checkpoint.last_checkpoint_path``: the
    checkpoint path is everything up to the first space, ``rest`` is the
    trailing timestamp (kept verbatim, including its leading space, or "").
    """
    stripped = line.rstrip("\n")
    ckpt_path, sep, rest = stripped.partition(" ")
    return ckpt_path, (sep + rest)


def detect_old_base(stored_path: str, new_base: Path) -> Optional[Path]:
    """Find the old base prefix of ``stored_path`` relative to ``new_base``.

    Returns the longest leading prefix ``old_base`` of ``stored_path`` such that
    swapping it for ``new_base`` yields an existing file -- i.e. the longest
    trailing suffix ``S`` where ``new_base / S`` is a file, with
    ``old_base = stored_path`` minus ``S``.

    Returns ``None`` when no suffix resolves to a file under ``new_base``.
    """
    parts = Path(stored_path).parts
    # Longest suffix first: a shorter suffix could spuriously match a same-named
    # file in an unrelated subtree, so prefer the most specific match.
    for k in range(len(parts), 0, -1):
        suffix = Path(*parts[-k:])
        if (new_base / suffix).is_file():
            old_parts = parts[: len(parts) - k]
            return Path(*old_parts) if old_parts else Path()
    return None


def find_fallback_ckpt(
    stored_path: str, new_base: Path, fallback_name: str
) -> Optional[Path]:
    """Locate a surviving checkpoint for ``stored_path`` under ``new_base``.

    Used when the exact checkpoint file is gone (e.g. only ``last.ckpt`` was
    kept). Maps the stored path's *directory* onto ``new_base`` -- the longest
    trailing suffix of that directory which exists as a directory under
    ``new_base`` -- and returns ``<that dir>/<fallback_name>`` if it is a file.

    Returns ``None`` when no matching directory holds ``fallback_name``.
    """
    parent_parts = Path(stored_path).parent.parts
    for k in range(len(parent_parts), 0, -1):
        cand_dir = new_base / Path(*parent_parts[-k:])
        if cand_dir.is_dir():
            candidate = cand_dir / fallback_name
            if candidate.is_file():
                return candidate
    return None


def rebase_line(ckpt_path: str, old_base: Path, new_base: Path) -> Optional[str]:
    """Return ``ckpt_path`` with ``old_base`` swapped for ``new_base``.

    Returns ``None`` if ``ckpt_path`` is not under ``old_base`` (left as-is).
    """
    path = Path(ckpt_path)
    try:
        suffix = path.relative_to(old_base)
    except ValueError:
        return None
    return str(new_base / suffix)


def plan_rewrite(
    lines: List[str], new_base: Path
) -> Tuple[Optional[Path], List[Tuple[int, str, str]]]:
    """Compute the old base and the per-line rewrites for one list file.

    Returns ``(old_base, changes)`` where ``changes`` is a list of
    ``(index, old_line_ckpt, new_line_ckpt)`` for lines whose path changes.
    ``old_base`` is ``None`` when the anchor (last non-blank) entry cannot be
    resolved under ``new_base``.
    """
    # Anchor on the last non-blank entry -- that is what restore_model reads.
    anchor_idx = next(
        (i for i in range(len(lines) - 1, -1, -1) if lines[i].strip()), None
    )
    if anchor_idx is None:
        return None, []

    anchor_ckpt, _ = parse_line(lines[anchor_idx])
    old_base = detect_old_base(anchor_ckpt, new_base)
    if old_base is None:
        return None, []

    changes: List[Tuple[int, str, str]] = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        ckpt, _ = parse_line(line)
        new_ckpt = rebase_line(ckpt, old_base, new_base)
        if new_ckpt is not None and new_ckpt != ckpt:
            changes.append((i, ckpt, new_ckpt))
    return old_base, changes


def apply_changes(lines: List[str], changes: List[Tuple[int, str, str]]) -> List[str]:
    """Return a new list of lines with the rewritten checkpoint paths applied."""
    new_lines = list(lines)
    for idx, _old_ckpt, new_ckpt in changes:
        _, rest = parse_line(lines[idx])
        new_lines[idx] = new_ckpt + rest
    return new_lines


def _plan_fallback(lines: List[str], new_base: Path, fallback_name: str) -> Optional[str]:
    """Return a new ``"<ckpt> <timestamp>"`` line for a surviving checkpoint.

    Called when the last entry's exact checkpoint is unresolvable. Returns the
    line to append, or ``None`` when no ``fallback_name`` is found.
    """
    anchor_ckpt, _ = parse_line(next(line for line in reversed(lines) if line.strip()))
    fallback = find_fallback_ckpt(anchor_ckpt, new_base, fallback_name)
    if fallback is None:
        return None
    return f"{fallback} {datetime.now()}"


def _report_plan(
    lines: List[str], new_base: Path, fallback_name: Optional[str]
) -> Tuple[bool, List[Tuple[int, str, str]], Optional[str]]:
    """Plan the rewrite and print diagnostics.

    Returns ``(ok, changes, append_line)``. ``ok`` is False only on a hard error
    (last checkpoint unresolvable and no fallback found). ``append_line`` is a
    new line to append (fallback mode); ``changes`` are in-place rewrites. Empty
    ``changes`` and ``None`` ``append_line`` with ``ok`` True means nothing to do.
    """
    old_base, changes = plan_rewrite(lines, new_base)
    if old_base is None:
        append_line = (
            _plan_fallback(lines, new_base, fallback_name) if fallback_name else None
        )
        if append_line is not None:
            print(
                "  Last checkpoint not found; appending surviving " f"'{fallback_name}':"
            )
            print(f"      + {append_line}")
            return True, [], append_line
        print(
            "  ERROR: could not locate the last checkpoint under "
            f"{new_base}.\n         The checkpoint is not beside this list "
            "file; leaving it untouched."
        )
        return False, [], None

    if not changes:
        print("  Paths already match this location -- nothing to do.")
        return True, [], None

    print(f"  Detected base swap:\n    OLD: {old_base}\n    NEW: {new_base}")
    print(f"  {len(changes)} line(s) to rewrite:")
    for idx, old_ckpt, new_ckpt in changes:
        exists = "" if Path(new_ckpt).is_file() else "  [!! still missing]"
        print(f"    line {idx + 1}:")
        print(f"      - {old_ckpt}")
        print(f"      + {new_ckpt}{exists}")
    return True, changes, None


def process_list_file(
    list_file: Path,
    dry_run: bool,
    assume_yes: bool,
    backup: bool,
    fallback_name: Optional[str] = None,
) -> bool:
    """Repair a single ``best_checkpoint.list``. Returns True on success."""
    list_file = list_file.resolve()
    print(f"\n=== {list_file} ===")
    if not list_file.is_file():
        print(f"  ERROR: not a file: {list_file}")
        return False

    new_base = list_file.parent
    lines = list_file.read_text(encoding="utf-8").splitlines()
    if not any(line.strip() for line in lines):
        print("  Empty checkpoint list -- nothing to do.")
        return True

    ok, changes, append_line = _report_plan(lines, new_base, fallback_name)
    if not ok:
        return False
    if not changes and append_line is None:
        return True

    if dry_run:
        print("  --dry-run: no changes written.")
    elif assume_yes or input("  Apply these changes? [y/N] ").strip().lower() in (
        "y",
        "yes",
    ):
        if backup:
            backup_path = list_file.with_name(list_file.name + ".bak")
            shutil.copy2(list_file, backup_path)
            print(f"  Backup written: {backup_path}")
        new_lines = apply_changes(lines, changes)
        if append_line is not None:
            new_lines.append(append_line)
        list_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        print(f"  Wrote {list_file}")
    else:
        print("  Skipped.")
    return True


def main():
    """main called from command line"""
    args = parse_arguments()
    all_ok = True
    for list_file in args.list_files:
        ok = process_list_file(
            list_file,
            dry_run=args.dry_run,
            assume_yes=args.yes,
            backup=not args.no_backup,
            fallback_name=args.fallback_ckpt,
        )
        all_ok = all_ok and ok
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
