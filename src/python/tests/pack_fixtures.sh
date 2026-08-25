#!/usr/bin/env bash
# Repack tests/fixtures/ into the committed fixtures.tar.zstd + fixtures.tar.index.
#
# Run this after adding, moving or regenerating anything under fixtures/ --
# the extracted tree is gitignored, so the tarball is the only committed copy.
#
#   bash pack_fixtures.sh            # repack from ./fixtures
#   bash pack_fixtures.sh --check    # verify the tarball matches ./fixtures, pack nothing
#
# Exclusions are listed once here so nobody has to remember them: everything
# excluded is REGENERATED at test time, and shipping it would ship stale state.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

FIXTURES_DIR="fixtures"
TAR="fixtures.tar"
ARCHIVE="${TAR}.zstd"
INDEX="${TAR}.index"

# Generated at test time, never shipped:
#  best_checkpoint.list  - written by pytest_sessionstart from best_checkpoint_template.list;
#                          holds absolute paths, so a shipped copy is wrong on every other machine
#  saccer3_2016-07/      - HDF5s extracted from saccer3_2016-07.tar.xz by the test fixtures
#  _smoke_hdf5.list      - scratch list written by smoke runs
#  mmap_cache/, *.npy    - lazy-loader signal caches
EXCLUDES=(
  --exclude=best_checkpoint.list
  --exclude=saccer3_2016-07
  --exclude=_smoke_hdf5.list
  --exclude=mmap_cache
  --exclude='*.npy'
  --exclude=__pycache__
  --exclude='.DS_Store'
)

if [[ ! -d "${FIXTURES_DIR}" ]]; then
  echo "ERROR: no '${FIXTURES_DIR}' directory here. Extract the archive first:" >&2
  echo "  zstd -dc ${ARCHIVE} | tar -xf -" >&2
  exit 1
fi

log_time() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$1"; }

if [[ "${1:-}" == "--check" ]]; then
  log_time "Comparing ${ARCHIVE} against ./${FIXTURES_DIR} ..."
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "${tmp_dir}"' EXIT
  zstd -dc "${ARCHIVE}" | tar -xf - -C "${tmp_dir}"
  if diff -rq "${tmp_dir}/${FIXTURES_DIR}" "${FIXTURES_DIR}" \
      --exclude=best_checkpoint.list \
      --exclude=saccer3_2016-07 \
      --exclude=_smoke_hdf5.list \
      --exclude=mmap_cache; then
    log_time "OK: archive matches the working fixtures tree."
  else
    log_time "DIFFERS: run 'bash pack_fixtures.sh' to repack."
    exit 1
  fi
  exit 0
fi

log_time "Packing ./${FIXTURES_DIR} ..."
tar "${EXCLUDES[@]}" -cf "${TAR}" "${FIXTURES_DIR}"

log_time "Writing ${INDEX} ..."
tar -tvf "${TAR}" > "${INDEX}"

log_time "Compressing to ${ARCHIVE} ..."
zstd -q -f -19 "${TAR}" -o "${ARCHIVE}"
rm -f "${TAR}"

log_time "Done: $(du -h "${ARCHIVE}" | cut -f1) archive, $(wc -l < "${INDEX}") entries."
