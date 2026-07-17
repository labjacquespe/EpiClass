#!/bin/bash
# Quarto post-render publish: mirror the rendered site into docs/, dropping the
# embedded-resource leftovers we don't ship (embed-resources inlines everything).
#
# The render directory (Quarto output-dir) is kept IN PLACE and mirrored, never
# moved. That is deliberate: moving it out empties it, so a subsequent
# single-section render would only contain that one section and would then wipe
# every other page out of docs/. Mirroring lets Quarto's incremental
# single-file renders accumulate on top of the full site, and keeps docs/ in
# sync without clobbering unrelated sections.
#
# Safe to re-run.

set -euo pipefail

# --- Resolve & validate script location ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
folder_name=$(basename "${SCRIPT_DIR}")

if [[ -z "${SCRIPT_DIR}" ]]; then
    echo "ERROR: Could not determine script directory." >&2
    exit 1
fi

if [[ "${folder_name}" != "epiclass-figures" ]]; then
    echo "ERROR: Script must run from 'epiclass-figures' (got: ${folder_name})." >&2
    exit 1
fi

# --- Validate the render directory (Quarto output-dir, kept in place) ---
render_path="${SCRIPT_DIR}/epiclass-paper"

if [[ ! -d "${render_path}" ]]; then
    echo "ERROR: Rendered output directory does not exist: ${render_path}" >&2
    exit 1
fi

# --- Resolve git root ---
git_root=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [[ -z "${git_root}" ]]; then
    echo "ERROR: Not inside a git repository; cannot resolve output location." >&2
    exit 1
fi

# --- Publish to docs/ ---
output_parent="${git_root}/docs"
output_folder="${output_parent}/epiclass-paper"

# Sanity guard: output_folder must be non-empty, not '/', not git_root, inside git_root
if [[ -z "${output_folder}" || "${output_folder}" == "/" || "${output_folder}" == "${git_root}" ]]; then
    echo "ERROR: Refusing to operate on suspicious output path: ${output_folder}" >&2
    exit 1
fi
case "${output_folder}" in
    "${git_root}"/*) ;;  # inside repo, OK
    *)
        echo "ERROR: Output folder is not inside git root: ${output_folder}" >&2
        exit 1
        ;;
esac

mkdir -p -- "${output_folder}"

# Mirror the rendered site into docs/. Trailing slashes copy directory contents.
#   --delete            prune files removed from the render (keeps docs/ in sync)
#   --delete-excluded   also drop the excluded leftovers if they linger in docs/
# Excludes are anchored to the transfer root so only the top-level copies are
# affected. site_libs/ is kept: the figure pages set embed-resources and don't
# need it, but index.html and about.html do NOT embed and load their navbar,
# search, theme and syntax-highlighting assets from site_libs/. We only drop
# resources/ and the source .qmd files copied alongside the figures.
echo "Mirroring ${render_path}/ -> ${output_folder}/"
rsync -a --delete --delete-excluded \
    --exclude='/resources/' \
    --exclude='/figs/*.qmd' \
    "${render_path}/" "${output_folder}/"

echo "Done."
