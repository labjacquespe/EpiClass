#!/bin/bash
#SBATCH --time=:::time::: # ex: 0:30:00
#SBATCH --account=:::your-account:::
#SBATCH --job-name=:::your-job-name:::
#SBATCH --output=./slurm_files/%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1 # GPU not needed for inference
#SBATCH --mem=:::memory::: # ex: 16G
#SBATCH --mail-user=:::your-email:::
#SBATCH --mail-type=END,FAIL
# shellcheck disable=SC1091  # Don't warn about sourcing unreachable files

# -- NOTE: The values above in between ':::' are to be replaced by the user --

# Ensemble prediction over the cross-validation fold models of one or more classifiers.
# predict_CV.py loads EVERY fold model of a CV run directory and scores the same input with
# each one, writing per-fold CSVs plus a concatenated long-format file with provenance.
#
# Usage:
#   <script>          # full run: every classifier listed below
#   <script> check    # validate paths, repair stale checkpoint lists, then exit
#
# Run 'check' once (on the login node) after copying or moving model directories: it is
# cheap, and it repairs the checkpoint lists that would otherwise fail the job at load time.

log_time() {
    echo "[$(date +%F_%T)] $1"
}

mode="${1:-}"
if [ "${mode}" == "check" ]; then
  log_time "Check mode enabled."
fi

log_time "Starting script."

export PYTHONUNBUFFERED=TRUE

# For HPC environments
module purge
ml StdEnv/2023 python/3.11

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "=========================================="
  echo "SLURM_JOB_ID = $SLURM_JOB_ID"
  echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
  echo "SLURM_TMPDIR = $SLURM_TMPDIR"
  echo "=========================================="
fi

gen_path="$HOME/:::project-path:::"
input_path="${gen_path}/epiclass/input"
output_path="${gen_path}/epiclass/output/logs"
gen_program_path="${gen_path}/sources/epiclass" # MODIFY: git root
program_path="${gen_program_path}/src/python/epiclass"

for path in ${gen_path} ${gen_program_path} ${input_path} ${output_path}; do
  if [ ! -d ${path} ]; then
    echo "${path} is not a directory. Please check the path." >&2
    exit 1
  else
    echo "Used directory: ${path}"
  fi
done

# Record which version of the code is being run.
cd ${gen_program_path} || { echo "Could not cd into '${gen_program_path}', exiting"; exit 1; }
echo "GIT COMMIT: $(git describe --always)"
cd - &> /dev/null


# --- choose source models + input files ---
# MODIFY THINGS HERE

release=":::release:::"   # ex: epiatlas-dfreeze-v2.1
assembly="hg38"
resolution="100kb"
basename="${resolution}_all_none"

base_log="${output_path}/${release}/${assembly}_${basename}"

# Each entry must be a CROSS-VALIDATION run directory: a folder holding the per-fold
# split*/ sub-directories, each with its own best_checkpoint.list. Do NOT point at a
# single split*/ directory -- that is what mains/predict.py is for (see example_predict.sh).
cv_subdir="10fold-oversampling" # MODIFY: name of the CV run dir holding split*/

assay_dir="${base_log}/assay_epiclass_1l_3000n/${cv_subdir}"
sex_dir="${base_log}/harmonized_donor_sex_1l_3000n/${cv_subdir}"

model_folders=( "${assay_dir}" "${sex_dir}" ) # MODIFY: one entry per classifier

# Input format. "single": one HDF5 per sample, needs a chrom sizes file, and predict_CV.py
# builds an mmap cache (point it at $SLURM_TMPDIR on HPC). "chunked": multi-sample HDF5
# chunks, self-contained -- no chrom sizes and no mmap cache.
input_format="single" # MODIFY: "single" or "chunked"

list_name=":::hdf5-list-name:::"  # single format: <name>.list under input/hdf5_list/
chunk_dir=":::chunk-dir:::"       # chunked format: directory of chunk_*.h5

out_subdir=":::output-subdir:::"  # ex: RNA_unstranded ; per-classifier output folder name


# --- Validating inputs ---

timestamp=$(date +%s)

log_time "Validating paths"

# Every model folder must be a CV run dir, i.e. hold at least one split*/best_checkpoint.list.
for path in "${model_folders[@]}"; do
  if [ ! -d ${path} ]; then
    echo "ERROR: ${path} is not a folder. Please check the path." >&2
    exit 1
  fi
  shopt -s nullglob
  fold_ckpts=( ${path}/split*/best_checkpoint.list )
  shopt -u nullglob
  if [ ${#fold_ckpts[@]} -eq 0 ]; then
    echo "ERROR: no split*/best_checkpoint.list under ${path}. Expected a CV run dir." >&2
    exit 1
  fi
  echo "INPUT MODEL (CV run, ${#fold_ckpts[@]} folds): ${path}"
done

# Assemble the input arguments for the chosen format.
if [ "${input_format}" == "chunked" ]; then
  hdf5_input="${chunk_dir}"
  extra_args="--chunked"
  if [ ! -d "${hdf5_input}" ]; then
    echo "ERROR: ${hdf5_input} is not a directory (chunked input)." >&2
    exit 1
  fi
  shopt -s nullglob
  chunk_files=( "${hdf5_input}"/chunk_*.h5 )
  shopt -u nullglob
  if [ ${#chunk_files[@]} -eq 0 ]; then
    echo "ERROR: no chunk_*.h5 under ${hdf5_input}." >&2
    exit 1
  fi
  echo "INPUT CHUNKS (${#chunk_files[@]} file(s)): ${hdf5_input}"
else
  hdf5_input="${input_path}/hdf5_list/${assembly}_${release}/${list_name}.list"
  chromsizes="${input_path}/chromsizes/hg38.noy.chrom.sizes" # MODIFY if needed
  extra_args="--chromsize ${chromsizes}"
  if [[ -n "$SLURM_TMPDIR" ]]; then
    extra_args="${extra_args} --mmap_dir ${SLURM_TMPDIR}/mmap_cache"
  fi
  for path in ${hdf5_input} ${chromsizes}; do
    if [ ! -f ${path} ]; then
      echo "ERROR: ${path} is not a file. Please check the path." >&2
      exit 1
    fi
    echo "Input: ${path}"
  done
fi


# --- Repair the checkpoint lists (check mode) ---
# best_checkpoint.list stores ABSOLUTE checkpoint paths, so copying or moving a model
# directory (or reading it from a different mount point) leaves every fold's list stale,
# and predict_CV.py fails to load the folds. Note that fold *discovery* still succeeds --
# it only checks that the list parses, not that the checkpoint exists -- so a stale list
# shows up as a load-time error per fold, not as an up-front failure.
#
# rebase_checkpoint_list.py detects the new base (the directory holding each list) and
# rewrites the stored paths in place, keeping a .bak copy of each original.
# --fallback-ckpt last.ckpt: when the recorded checkpoint is gone entirely (CV runs that
# kept only last.ckpt and deleted the per-epoch best checkpoints), it appends a new line
# pointing at the surviving last.ckpt, which restoration reads.
# It is pure-stdlib, so it runs under the module python before any venv is activated.
# Drop --yes and add --dry-run to preview the rewrite instead of applying it.
if [ "${mode}" == "check" ]; then
  log_time "Rebasing checkpoint lists onto current location (if needed)"
  shopt -s nullglob
  ckpt_lists=()
  for path in "${model_folders[@]}"; do
    ckpt_lists+=( ${path}/split*/best_checkpoint.list )
  done
  shopt -u nullglob
  if [ ${#ckpt_lists[@]} -gt 0 ]; then
    python ${program_path}/utils/rebase_checkpoint_list.py "${ckpt_lists[@]}" \
      --yes --fallback-ckpt last.ckpt
  fi
fi

# Preconditions passed, copy launch script to log dir for reproducibility.
if [[ -n "$SLURM_JOB_ID" ]]; then
  scontrol write batch_script ${SLURM_JOB_ID} ./slurm_files/launch_script_${SLURM_JOB_NAME}-job${SLURM_JOB_ID}.sh
fi

if [ "${mode}" == "check" ]; then
  log_time "Check finished successfully."
  exit 0
fi


# --- use correct environment ---

set -e # exit on error
if [[ -n "$SLURM_JOB_ID" ]]; then
  # create venv on the fly
  cd $SLURM_TMPDIR
  python -m venv epiclass_env
  source epiclass_env/bin/activate
  python ${gen_program_path}/install.py &> job${SLURM_JOB_ID}_venv_setup.log
  cd - &> /dev/null
else
  source /path/to/preinstalled/venv/bin/activate # MODIFY
fi


# --- launch ---

log_time "Main script launch"

# predict_CV.py takes the input first, then the CV run directory:
#   predict_CV.py <hdf5 list | chunk dir> <cv_root> [--chunked] [--chromsize ...]
# --output_dir defaults to <cv_root>/predictionsCV when omitted.
for model_folder in "${model_folders[@]}"; do

  out_dir="${model_folder}/predictionsCV/${out_subdir}"
  mkdir -p ${out_dir}
  out1="${out_dir}/output_job${SLURM_JOB_ID}_${SLURM_JOB_NAME}_${timestamp}.o"
  out2="${out_dir}/output_job${SLURM_JOB_ID}_${SLURM_JOB_NAME}_${timestamp}.e"

  cmd="python ${program_path}/mains/predict_CV.py ${hdf5_input} ${model_folder} ${extra_args} --output_dir ${out_dir} --batch_size 1024"
  printf '\n%s\n' "Launching following command"
  printf '%s\n' "${cmd} >> ${out1} 2>> ${out2}"
  ${cmd} >>${out1} 2>>${out2}

done

log_time "Done."


# -- You could then augment the prediction files with metadata, if it is known --

# to_augment="${out_dir}/:::concatenated-prediction-file:::"
# metadata="${input_path}/metadata/${assembly}_${release}_metadata.json"

# printf '\n%s\n' "Launching following command"
# printf '%s\n' "python ${program_path}/utils/augment_predict_file.py ${to_augment} ${metadata} --all-categories"
# python ${program_path}/utils/augment_predict_file.py ${to_augment} ${metadata} --all-categories
