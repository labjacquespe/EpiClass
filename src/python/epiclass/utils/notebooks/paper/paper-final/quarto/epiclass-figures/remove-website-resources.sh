#!/bin/bash
# Quarto post-render cleanup: prune embedded-resource leftovers and publish to docs/.
# Safe to re-run; verifies working directory before destructive operations.

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

# --- Validate generated docs path ---
docs_path="${SCRIPT_DIR}/epiclass-paper"

if [[ ! -d "${docs_path}" ]]; then
    echo "ERROR: Generated docs directory does not exist: ${docs_path}" >&2
    exit 1
fi

# --- Cleanup unnecessary resources (we use embed-resources) ---
resources_path="${docs_path}/resources"
if [[ -d "${resources_path}" ]]; then
    echo "Removing ${resources_path}"
    rm -rf -- "${resources_path}"
fi

# Remove .qmd files from figs/ (nullglob so no-match isn't an error)
figs_path="${docs_path}/figs"
if [[ -d "${figs_path}" ]]; then
    shopt -s nullglob
    qmd_files=( "${figs_path}"/*.qmd )
    shopt -u nullglob
    if (( ${#qmd_files[@]} > 0 )); then
        echo "Removing ${#qmd_files[@]} .qmd file(s) from ${figs_path}"
        rm -f -- "${qmd_files[@]}"
    fi
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

mkdir -p -- "${output_parent}"

if [[ -e "${output_folder}" ]]; then
    echo "Removing existing ${output_folder}"
    rm -rf -- "${output_folder}"
fi

echo "Moving generated docs from ${docs_path} to ${output_folder}"
mv -- "${docs_path}" "${output_folder}"

echo "Done."
