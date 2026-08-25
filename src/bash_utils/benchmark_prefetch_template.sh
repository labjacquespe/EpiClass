#!/bin/bash
#SBATCH --time=:::time:::
#SBATCH --account=:::your-account:::
#SBATCH --job-name=:::your-job-name:::
#SBATCH --output=./slurm_files/%x-job%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=:::memory::: # ex: 16G -- see the note on the data>RAM lever below
#SBATCH --mail-user=:::your-email:::
#SBATCH --mail-type=END,FAIL
# shellcheck disable=SC1091  # Don't warn about sourcing unreachable files

# -- NOTE: The values above in between ':::' are to be replaced by the user --
#
# Request the MAXIMUM number of cores you want to test once (--cpus-per-task=4).
# The benchmark sweeps SUBSETS of them internally by capping CPU affinity per
# run, so you do NOT launch one job per core count.
#
# Two ways to force the realistic "data does not fit in RAM" regime:
#   1. In-script: the benchmark evicts the dataset mmap page cache
#      (posix_fadvise) before each timed epoch -- always on. No action needed.
#   2. cgroup lever: set --mem BELOW the mmap dataset size (printed as
#      `mmap_bytes` in the results) so the job cgroup evicts page cache under
#      natural memory pressure. Use this to model "partially fits in RAM".

set -e # exit on error

log_time() {
    echo "Time: $(date +%F_%T) - $1"
}

export PYTHONUNBUFFERED=TRUE

# For HPC environments
module purge
ml StdEnv/2023 python/3.11
export FLEXIBLAS=openblas # BLIS problem in StdEnv/2023, gets hit on Narval

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "=========================================="
  echo "SLURM_JOB_ID = $SLURM_JOB_ID"
  echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
  echo "=========================================="
fi

gen_path="$HOME/:::project-path:::" # MODIFY, input/output directories
input_path="${gen_path}/epiclass/input"
output_path="${gen_path}/epiclass/output/benchmark"

gen_program_path="${gen_path}/sources/epiclass" # MODIFY: git root
program_path="${gen_program_path}/src/python/epiclass"

slurm_out_folder="${gen_path}/epiclass/output/sub/slurm_files"

for path in ${slurm_out_folder} ${gen_program_path} ${input_path}; do
  if [ ! -d ${path} ]; then
    echo "${path} is not a directory. Please check the path."
    exit 1
  else
    echo "Used directory: ${path}"
  fi
done


# --- choose benchmark config + output ---

# MODIFY THINGS HERE.
# The config JSON holds the data paths, the parameter sweep, and the run
# settings (see input-format/benchmark_prefetch.json for the schema).
config="${input_path}/benchmark/benchmark_prefetch.json" # IMPORTANT
timestamp=$(date +%s)
logdir="${output_path}/prefetch_${timestamp}"

for path in ${config}; do
  if [ ! -f ${path} ]; then
    echo "${path} is not a file. Please check the path."
    exit 1
  else
    echo "Input: ${path}"
  fi
done


# --- use correct environment ---

if [[ -n "$SLURM_JOB_ID" ]]; then
  # create venv on the fly
  log_time "Starting venv setup"
  cd $SLURM_TMPDIR
  python -m venv epiclass_env
  source epiclass_env/bin/activate
  python ${gen_program_path}/install.py &> job${SLURM_JOB_ID}_venv_setup.log
  log_time "Venv setup done"
else
  source /path/to/preinstalled/venv/bin/activate # MODIFY
fi


# --- MAIN PROGRAM ---

cd ${program_path}
mkdir -p ${logdir}

log_time "Launching benchmark"
printf '\n%s\n' "Launching following command"
printf '%s\n' "python -m epiclass.utils.benchmark.benchmark_prefetch --config ${config} --logdir ${logdir}"
python -m epiclass.utils.benchmark.benchmark_prefetch --config ${config} --logdir ${logdir}
log_time "Benchmark done"

# Copy slurm output file to log dir
if [[ -n "$SLURM_JOB_ID" ]]; then
  slurm_out_file="${SLURM_JOB_NAME}-*${SLURM_JOB_ID}.out"
  cp -v ${slurm_out_folder}/${slurm_out_file} ${logdir}/ || true
fi
