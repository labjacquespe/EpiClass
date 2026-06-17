#!/bin/bash
#SBATCH --account=:::your-account:::
#SBATCH --job-name=:::your-job-name:::
#SBATCH --output=%x-%j.out
#SBATCH --time=:::time::: # ex: 48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=:::memory::: # ex: 16G
#SBATCH --mail-user=:::your-email:::
#SBATCH --mail-type=END,FAIL
# shellcheck disable=SC1091  # Don't warn about sourcing unreachable files

# -- NOTE: The values above in between ':::' are to be replaced by the user. --
#
# Create single-sample HDF5 signal files from bigWigs with epigeec, the input
# format consumed by EpiClass's LazyHdf5Loader.
#
# For each bigWig, 'epigeec to_hdf5' builds the first (finest) resolution; any
# coarser resolutions requested are derived from it with
# 'epigeec-converter down-resolution' (the bigWig is read only once). Work runs in
# parallel (GNU parallel) into node-local $SLURM_TMPDIR, then is rsynced to the
# output dir; the run is resumable (samples whose outputs already exist are
# skipped) and saves partial work if killed (e.g. walltime).
#
# Runs under SLURM (builds an epigeec venv from a wheelhouse) or locally (uses
# epigeec already on PATH). Cluster-specific defaults are marked "EDIT" below and
# can also be overridden per-run via -C / -L / -o.
#
# Examples:
#   sbatch --job-name=hdf5_1kb make_hdf5_template.sh -d /path/to/bw -r 1000 -c noy
#   sbatch make_hdf5_template.sh -l samples.list -r 1000,10000,100000 -c can -o /out/dir

set -e

# --- CLI --- #

usage() {
    echo "Usage: $0 [-h] [-t] -r RESOLUTIONS -c CHROM [-C CHROMDIR] [-L LOGDIR] [-o OUTBASE] [-n NAME] [-j JOBTAG] (-d DIR | -l LIST)"
    echo "  -h            Help. Display this message and quit."
    echo "  -t            Test. Only process the first two files."
    echo "  -r RESOLUTIONS Required. Comma-separated bin sizes in bp."
    echo "                 The first is built from the bigWig with 'epigeec to_hdf5';"
    echo "                 each subsequent one is derived from the previous with"
    echo "                 'epigeec-converter down-resolution' (bigWig read once)."
    echo "                 Multiple values must be ascending, each a multiple of the"
    echo "                 previous, e.g. -r 1000,10000,100000."
    echo "  -c CHROM      Required. Chromsize name; selects hg38.CHROM.chrom.sizes and"
    echo "                tags output filenames. Fails early if no such file exists."
    echo "  -C CHROMDIR   Directory holding the hg38.CHROM.chrom.sizes files."
    echo "                Defaults to \${gen_path}/epiclass/input/chromsizes."
    echo "  -L LOGDIR     Directory for GNU parallel joblogs / failure lists."
    echo "                Defaults to \${gen_path}/epiclass/output/sub/slurm_files."
    echo "  -o OUTBASE    Output base directory; results go to OUTBASE/NAME/RESOLUTION/."
    echo "                For a single -r resolution the RESOLUTION subdir is dropped"
    echo "                (OUTBASE/NAME is the literal target). Defaults to \${epiatlas_dir}/hdf5."
    echo "  -n NAME       Optional subdirectory under OUTBASE (then /RESOLUTION)."
    echo "                Defaults to the basename of -d DIR. With -l it is optional"
    echo "                when -o is given, but required with the default OUTBASE."
    echo "  -j JOBTAG     Prefix for joblog / failure-list filenames, to tell runs apart."
    echo "                Defaults to the SLURM job name. NOTE: to rename the actual"
    echo "                SLURM job + its %x stdout file, pass 'sbatch --job-name=...'."
    echo "  -d DIR        Input mode: directory containing .bw files to process."
    echo "  -l LIST       Input mode: file listing full paths to .bw files (one per line)."
    echo "Exactly one of -d or -l is required."
    exit
}

test_mode=""
input_dir=""
input_list=""
resolution_arg=""
chrom_name=""
chromsize_dir=""
log_folder=""
out_base=""
out_name=""
job_tag=""
while getopts "htr:c:C:L:o:n:j:d:l:" optchar; do
    case "${optchar}" in
        h) usage ;;
        t) test_mode=true ;;
        r) resolution_arg="${OPTARG}" ;;
        c) chrom_name="${OPTARG}" ;;
        C) chromsize_dir="${OPTARG}" ;;
        L) log_folder="${OPTARG}" ;;
        o) out_base="${OPTARG}" ;;
        n) out_name="${OPTARG}" ;;
        j) job_tag="${OPTARG}" ;;
        d) input_dir="${OPTARG}" ;;
        l) input_list="${OPTARG}" ;;
        *) usage ;;
    esac
done

if [[ -n "${input_dir}" && -n "${input_list}" ]]; then
    echo "Provide only one of -d (directory) or -l (filelist), not both. Exiting."
    exit 1
fi
if [[ -z "${input_dir}" && -z "${input_list}" ]]; then
    echo "No input given. Provide -d DIR or -l LIST."
    usage
fi
if [[ -z "${resolution_arg}" ]]; then
    echo "No resolution given. Provide -r RESOLUTIONS (e.g. -r 1000,10000,100000)."
    usage
fi
if [[ -z "${chrom_name}" ]]; then
    echo "No chromsize name given. Provide -c CHROM (e.g. -c noy)."
    usage
fi

# -- Parse and validate the resolution list --
# res_sizes holds the requested bin sizes (bp). The first is created directly
# from the bigWig; the rest are down-resolution-derived from the previous one,
# which requires them ascending and each a multiple of the previous.
IFS=',' read -ra res_sizes <<< "${resolution_arg}"
for s in "${res_sizes[@]}"; do
  if ! [[ "${s}" =~ ^[0-9]+$ ]]; then
    echo "Resolutions must be positive integers (bp). Got: '${s}' in '${resolution_arg}'. Exiting."
    exit 1
  fi
done
for ((i = 1; i < ${#res_sizes[@]}; i++)); do
  prev="${res_sizes[i - 1]}"
  cur="${res_sizes[i]}"
  if (( cur <= prev || cur % prev != 0 )); then
    echo "Resolutions must be ascending and each a multiple of the previous"
    echo "(for down-resolution chaining). Got: '${resolution_arg}'. Exiting."
    exit 1
  fi
done

# Output subdirectory name (under OUTBASE): explicit -n wins; otherwise default to
# the -d directory basename. With -l there is no directory to infer from: that is
# fine if -o was given (OUTBASE is then the full target), but with the default
# OUTBASE we require -n so unrelated runs don't all dump into the same hdf5 dir.
if [[ -z "${out_name}" ]]; then
  if [[ -n "${input_dir}" ]]; then
    out_name="$(basename "${input_dir%/}")"
  elif [[ -z "${out_base}" ]]; then
    echo "With -l and the default output base, an output name is required:"
    echo "pass -n NAME, or give an explicit -o OUTBASE. Exiting."
    exit 1
  fi
fi

# Load cluster modules when an HPC module system is present; skipped when running
# locally (no 'module' command).
if command -v module &>/dev/null; then
  module purge
  ml StdEnv/2023 python/3.11
fi

log_time() {
    echo "Time: $(date +%F_%T) - $1"
}

# Human-readable label for a resolution (e.g. 1000 -> 1kb, 100000 -> 100kb).
res_label_of() {  # $1=size in bp
  local s="$1"
  if (( s % 1000000 == 0 )); then
    echo "$((s / 1000000))mb"
  elif (( s % 1000 == 0 )); then
    echo "$((s / 1000))kb"
  else
    echo "${s}bp"
  fi
}

# -- Initial setup --

log_time "Setting up paths."

# === EDIT: cluster-specific path defaults ===================================
# These define the default locations for chromsizes (-C), joblogs (-L), and
# output (-o) when those flags are not passed. Point them at your own layout.
gen_path="/lustre06/project/6007017/rabyj"
scratch_dir="/lustre07/scratch/rabyj"
epiatlas_dir="${scratch_dir}/local_ihec_data/epiatlas"
# ===========================================================================

# Apply defaults for any path not overridden on the CLI (-C/-L/-o).
log_folder="${log_folder:-${gen_path}/epiclass/output/sub/slurm_files}"
chromsize_dir="${chromsize_dir:-${gen_path}/epiclass/input/chromsizes}"
out_base="${out_base:-${epiatlas_dir}/hdf5}"

# Checking directories (out_base is created with mkdir -p, so it need not pre-exist)
for path in "${log_folder}" "${chromsize_dir}"; do
  if [[ ! -d "${path}" ]]; then
    echo "${path} is not a directory. Please check the path. Exiting."
    exit 1
  fi
  echo "Using directory: ${path}"
done

# -- Resolutions --
# Build parallel arrays indexed in lockstep with res_sizes.

dset_name="${chrom_name}"

# Output root: OUTBASE, plus the optional NAME subdir when one was set/derived.
if [[ -n "${out_name}" ]]; then
  out_root="${out_base}/${out_name}"
else
  out_root="${out_base}"
fi

# Per-resolution subdir keeps multiple resolutions apart. For a single resolution
# it is redundant (the filename already carries the resolution tag), so outputs go
# straight into out_root and OUTBASE can be the literal target directory.
single_res=0
(( ${#res_sizes[@]} == 1 )) && single_res=1

res_labels=()
final_dirs=()
for s in "${res_sizes[@]}"; do
  label="$(res_label_of "${s}")"
  res_labels+=("${label}")
  if (( single_res )); then
    d="${out_root}"
  else
    d="${out_root}/${label}"
  fi
  mkdir -p "${d}"
  final_dirs+=("${d}")
done

# Helper: final hdf5 path for a given sample name and resolution index
final_path() {  # $1=sample_name  $2=res_index
  echo "${final_dirs[$2]}/${1}_${res_labels[$2]}_${dset_name}.hdf5"
}

# Checking files
chromsize="${chromsize_dir}/hg38.${chrom_name}.chrom.sizes"
if [[ ! -f "${chromsize}" ]]; then
  echo "${chromsize} does not exist (check -c '${chrom_name}'). Exiting"
  exit 1
fi
echo "Using file: ${chromsize}"

# -- Resolve input into a single full-path bigWig list --
# Both input modes (-d directory / -l filelist) are normalized here to one list
# of full bigWig paths (one per line), used everywhere downstream.
if [[ -n "${input_dir}" ]]; then
  if [[ ! -d "${input_dir}" ]]; then
    echo "${input_dir} does not exist or is not a directory. Exiting."
    exit 1
  fi
  bw_list="tmp_${out_name}_input_bw.list"
  find "${input_dir}" -type f -name '*.bw' | sort > "${bw_list}"
  echo "Input mode: directory '${input_dir}' -> $(wc -l < "${bw_list}") bigWigs"
else
  if [[ ! -f "${input_list}" ]]; then
    echo "${input_list} does not exist. Exiting."
    exit 1
  fi
  bw_list="${input_list}"
  echo "Input mode: filelist '${input_list}'"
fi

# -- Count files --
# A sample counts as "done" only when all requested resolutions exist.

total_input=$(wc -l < "${bw_list}")

echo "=========================================="
echo "Resolutions:          ${resolution_arg} bp"
echo "Total input files:    ${total_input}"
for i in "${!res_labels[@]}"; do
  done_res=$(find "${final_dirs[$i]}" -name "*_${res_labels[$i]}_${dset_name}.hdf5" -type f | wc -l)
  printf 'Completed %-6s       %s\n' "${res_labels[$i]}:" "${done_res}"
done
echo "=========================================="

# -- Run mode: SLURM vs local --
# Under SLURM we use the scheduler-provided scratch dir and resources. Locally we
# fall back to a temp workdir and the host CPU count (and skip the cluster venv
# setup below, relying on epigeec already on PATH). Either way, the resource paths
# checked above (log_folder, chromsize) must exist.
if [[ -n "${SLURM_JOB_ID}" ]]; then
  in_slurm=1
  echo "=========================================="
  echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
  echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST}"
  echo "SLURM_TMPDIR = ${SLURM_TMPDIR}"
  echo "=========================================="
else
  in_slurm=0
  : "${SLURM_JOB_ID:=local-$$}"
  : "${SLURM_JOB_NAME:=make_hdf5}"
  : "${SLURM_JOB_NODELIST:=localhost}"
  : "${SLURM_CPUS_PER_TASK:=$(nproc)}"
  SLURM_TMPDIR="$(mktemp -d)"
  log_time "Not running under SLURM - local mode (workdir: ${SLURM_TMPDIR}, cpus: ${SLURM_CPUS_PER_TASK})."
fi

# Prefix for joblog / failure-list filenames; defaults to the (SLURM) job name.
job_tag="${job_tag:-${SLURM_JOB_NAME}}"

# Per-resolution temporary (node-local) output directories
tmp_base="${SLURM_TMPDIR}/epigeec_hdf5"
echo "Working tmpdir (ssh to ${SLURM_JOB_NODELIST} to inspect): ${tmp_base}"
tmp_dirs=()
for label in "${res_labels[@]}"; do
  d="${tmp_base}/${label}"
  mkdir -p "${d}"
  tmp_dirs+=("${d}")
done

# -- Tooling: epigeec (+ epigeec-converter when chaining) --
# Under SLURM, build a fresh venv from the wheelhouse. Locally, use whatever is
# already on PATH and just verify it is there.
if [[ ${in_slurm} -eq 1 ]]; then
  env_name="epigeec"
  log_time "Setting up venv '${env_name}'"

  # === EDIT: cluster-specific tooling sources ==============================
  # uv config + the epigeec / epigeec-converter install sources for your site.
  export UV_CONFIG_FILE="$HOME/.config/uv/compute.toml"
  epigeec_wheel="$HOME/wheelhouse/epigeec*.whl"
  epigeec_converter_src="/lustre06/project/6007017/rabyj/sources/epigeec_converter"
  # =========================================================================

  # Build the venv via an absolute path rather than cd-ing: changing the working
  # directory would break any relative -d/-l input path (and relative entries in a
  # filelist), since those are resolved later when 'parallel' runs epigeec.
  venv_dir="${SLURM_TMPDIR}/${env_name}"
  uv venv -p=3.11 --seed --clear "${venv_dir}"
  source "${venv_dir}/bin/activate"
  # shellcheck disable=SC2086  # intentional glob on the wheel path
  uv pip install ${epigeec_wheel}
  # epigeec-converter is only needed when chaining to coarser resolutions.
  if (( ${#res_sizes[@]} > 1 )); then
    uv pip install "${epigeec_converter_src}"
  fi
else
  log_time "Local mode: using tools already on PATH."
  if ! command -v epigeec &>/dev/null; then
    echo "epigeec not found on PATH. Activate the environment that provides it. Exiting."
    exit 1
  fi
  if (( ${#res_sizes[@]} > 1 )) && ! command -v epigeec-converter &>/dev/null; then
    echo "epigeec-converter not found on PATH (needed for multi-resolution chaining). Exiting."
    exit 1
  fi
fi

nb_cores=$SLURM_CPUS_PER_TASK
nb_jobs=$((nb_cores * 2))

# -- Build work list --
# Process a sample if ANY of its requested resolutions is missing. The full chain
# is regenerated for those samples.

log_time "Building work list"

# In test mode, only consider the first two files from the list.
work_source="${bw_list}"
if [[ -n "${test_mode}" ]]; then
  work_source="${SLURM_TMPDIR}/test_bw.list"
  head -n2 "${bw_list}" > "${work_source}"
  log_time "TEST MODE: limiting to first $(wc -l < "${work_source}") files from the list."
fi

# all_files_list holds full bigWig paths; the sample name (used for output
# filenames and resume checks) is the basename without the .bw extension.
all_files_list="${SLURM_TMPDIR}/all_files.txt"
while read -r bw_path; do
  sample_name="$(basename "${bw_path}" .bw)"
  for i in "${!res_labels[@]}"; do
    if [[ ! -f "$(final_path "${sample_name}" "${i}")" ]]; then
      echo "${bw_path}"
      break
    fi
  done
done < "${work_source}" > "${all_files_list}"

total_files=$(wc -l < "${all_files_list}")
log_time "Total samples to process: ${total_files}"

if [[ ${total_files} -eq 0 ]]; then
  log_time "No files to process. Exiting."
  exit 0
fi

# -- Flush helper --
# Outputs are written to node-local tmp for speed, then copied to the persistent
# final dir(s). Called explicitly at the end (normal path); the trap re-runs it if
# the job is killed early (walltime/error) so partial work survives. The 'synced'
# guard makes it idempotent so the two paths never double-copy.
synced=0
flush_to_final() {
  [[ ${synced} -eq 1 ]] && return
  synced=1
  log_time "Flushing tmp results to final dir(s)..."
  for i in "${!res_labels[@]}"; do
    rsync -a "${tmp_dirs[$i]}/" "${final_dirs[$i]}/" --include='*.hdf5' --exclude='*'
  done
  log_time "Flush complete."
  # SLURM auto-reclaims $SLURM_TMPDIR; in local mode we created it, so clean it up.
  if [[ ${in_slurm} -eq 0 ]]; then
    rm -rf "${SLURM_TMPDIR}"
  fi
}
# shellcheck disable=SC2329  # invoked indirectly via the EXIT trap
trap flush_to_final EXIT  # safety net if the job is killed before the explicit call

# Helper: count failed jobs (exit code != 0) in a GNU parallel joblog
count_failures() {  # $1=joblog
  awk -F'\t' 'NR>1 && $7 != 0' "$1" | wc -l
}

# -- Main script --

n_res=${#res_sizes[@]}
total_failed=0

# Stage 1: bigWig -> first resolution
log_time "Stage 1/${n_res}: epigeec to_hdf5 (${res_sizes[0]} bp)"
joblog="${log_folder}/${job_tag}_job${SLURM_JOB_ID}_to_hdf5_${res_labels[0]}.log"
parallel_exit=0
# {} is the full bigWig path; {/.} is its basename without the .bw extension.
parallel --joblog "${joblog}" -j "${nb_jobs}" \
  epigeec to_hdf5 -bw "{}" "${chromsize}" "${res_sizes[0]}" \
    "${tmp_dirs[0]}/{/.}_${res_labels[0]}_${dset_name}.hdf5" \
  :::: "${all_files_list}" || parallel_exit=$?

if [[ ${parallel_exit} -ne 0 ]]; then
  failed_n=$(count_failures "${joblog}")
  total_failed=$((total_failed + failed_n))
  log_time "WARNING: ${failed_n} jobs failed in stage 1 (joblog: ${joblog})"
  failed_list="${log_folder}/${job_tag}_job${SLURM_JOB_ID}_failed.txt"
  awk -F'\t' 'NR>1 && $7 != 0' "${joblog}" | \
    grep -oP '(?<=-bw )\S+' > "${failed_list}"
  log_time "Failed bigWigs logged to: ${failed_list}"
fi

# Stages 2..n: down-resolution chain. Each derives from the previous resolution;
# samples whose source is missing (e.g. failed stage 1) are logged by GNU parallel
# as failures and skipped.
for ((i = 1; i < n_res; i++)); do
  src=$((i - 1))
  log_time "Stage $((i + 1))/${n_res}: epigeec-converter down-resolution (${res_sizes[$i]} bp)"
  joblog="${log_folder}/${job_tag}_job${SLURM_JOB_ID}_down_${res_labels[$i]}.log"
  parallel_exit=0
  parallel --joblog "${joblog}" -j "${nb_jobs}" \
    epigeec-converter down-resolution \
      "${tmp_dirs[$src]}/{/.}_${res_labels[$src]}_${dset_name}.hdf5" \
      "${res_sizes[$i]}" \
      "${tmp_dirs[$i]}/{/.}_${res_labels[$i]}_${dset_name}.hdf5" \
    :::: "${all_files_list}" || parallel_exit=$?

  if [[ ${parallel_exit} -ne 0 ]]; then
    failed_n=$(count_failures "${joblog}")
    total_failed=$((total_failed + failed_n))
    log_time "WARNING: ${failed_n} jobs failed in down-resolution to ${res_labels[$i]} (joblog: ${joblog})"
  fi
done

# Copy node-local results to their persistent home (also runs via trap if killed).
flush_to_final

# Final summary
if [[ ${total_failed} -gt 0 ]]; then
  log_time "Job complete with errors. ${total_failed} stage-jobs failed across ${n_res} resolution(s) for ${total_files} samples."
  exit 1
else
  log_time "Job complete. Processed ${total_files} samples at ${n_res} resolution(s) successfully."
  exit 0
fi
