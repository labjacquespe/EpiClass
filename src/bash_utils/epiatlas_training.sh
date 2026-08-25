#!/bin/bash
#SBATCH --time=:::time:::
#SBATCH --account=:::your-account:::
#SBATCH --job-name==:::your-job-name:::
#SBATCH --output=./slurm_files/%x-job%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=:::memory::: # ex: 16G
#SBATCH --mail-user=:::your-email:::
#SBATCH --mail-type=END,FAIL
# shellcheck disable=SC1091  # Don't warn about sourcing unreachable files

# -- NOTE: The values above in between ':::' are to be replaced by the user --

set -e # exit on error

log_time() {
    echo "Time: $(date +%F_%T) - $1"
}

export PYTHONUNBUFFERED=TRUE
# export UV_CONFIG_FILE="$HOME/.config/uv/compute.toml"

# For HPC environments
module purge
ml StdEnv/2023 python/3.11
ml httpproxy # for comet-ml

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "=========================================="
  echo "SLURM_JOB_ID = $SLURM_JOB_ID"
  echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
  echo "=========================================="
fi

gen_path="$HOME/:::project-path:::" # MODIFY, input/output directories
input_path="${gen_path}/epiclass/input"
output_path="${gen_path}/epiclass/output/logs"

gen_program_path="${gen_path}/sources/epiclass" # MODIFY: git root
program_path="${gen_program_path}/src/python/epiclass"

slurm_out_folder="${gen_path}/epiclass/output/sub/slurm_files"

for path in ${slurm_out_folder} ${gen_program_path} ${input_path} ${output_path}; do
  if [ ! -d ${path} ]; then
    echo "${path} is not a directory. Please check the path."
    exit 1
  else
    echo "Used directory: ${path}"
  fi
done


# --- choose category + hparams + source files ---

# MODIFY THINGS HERE

# RESTORE="--restore" # COMMENT IF TRAINING # IMPORTANT
# NO_VALID="hell yeah" # COMMENT IF 10fold  TRAINING # IMPORTANT

category="assay_epiclass"

export EXCLUDE_LIST='["other", "--", "NA", "", "unknown"]'
export MIN_CLASS_SIZE="10" # IMPORTANT

hparams="human_longer_oversample" # IMPORTANT

release="epiatlas-dfreeze-v2.1"
assembly="hg38"

resolution="100kb"
basename="${resolution}_all_none"
list_name="${basename}" # IMPORTANT

export LAYER_SIZE="3000" # IMPORTANT
export NB_LAYER="1"

log="${output_path}/${release}/${assembly}_${basename}/${category}_${NB_LAYER}l_${LAYER_SIZE}n" # IMPORTANT# IMPORTANT# IMPORTANT# IMPORTANT
log="${log}/10fold-oversampling"


# --- Creating correct paths for programs/launching ---

timestamp=$(date +%s)

hparams="${input_path}/hparams/${hparams}.json"
hdf5_list="${input_path}/hdf5_list/hg38_epiatlas-freeze-v2/${list_name}.list"
chroms="${input_path}/chromsizes/hg38.noy.chrom.sizes"
metadata="${input_path}/metadata/dfreeze-v2/hg38_2023-epiatlas-dfreeze-pospurge-nodup.json"
out1="${log}/output_job${SLURM_JOB_ID}_${SLURM_JOB_NAME}_${timestamp}.o"
out2="${log}/output_job${SLURM_JOB_ID}_${SLURM_JOB_NAME}_${timestamp}.e"

for path in ${hparams} ${hdf5_list} ${chroms} ${metadata}; do
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


# --- Pre-checks ---

cd ${program_path}

# This script is particularly useful when working on HPC, because it sets sticky permissions properly, so new files stay within their project group.
printf '\n%s\n' "Launching following command"
printf '%s\n' "python ${program_path}/utils/check_dir.py ${log}"
python ${program_path}/utils/check_dir.py ${log}

# Preconditions passed, copy launch script to log dir.
if [[ -n "$SLURM_JOB_ID" ]]; then
  scontrol write batch_script ${SLURM_JOB_ID} ${log}/launch_script_${SLURM_JOB_NAME}-job${SLURM_JOB_ID}.sh
fi


# --- MAIN PROGRAM ---

log_time "Launching training"
printf '\n%s\n' "Launching following command"
if [[ -n "$NO_VALID" ]]; then #if variable exists
  # --- complete training without validation set launch ---
  if [[ "$log" == *"10fold"* ]]; then
    log="$log/notactually10foldbaka"
    printf '\n%s\n' "Incoherent log path, changing log to $log"
  fi

  printf '%s\n' "python ${program_path}/epiatlas_training_no_valid.py $category ${hparams} ${hdf5_list} ${chroms} ${metadata} ${log} > ${out1} 2> ${out2}"
  python ${program_path}/epiatlas_training_no_valid.py $category ${hparams} ${hdf5_list} ${chroms} ${metadata} ${log} >"${out1}" 2>"${out2}"
  log_time "Training done"
  exit

elif [[ -n "$RESTORE" ]]; then
  # --- kfold launch ---
  printf '%s\n' "python ${program_path}/epiatlas_training.py $category ${hparams} ${hdf5_list} ${chroms} ${metadata} ${log} --restore > ${out1} 2> ${out2}"
  python ${program_path}/epiatlas_training.py $category ${hparams} ${hdf5_list} ${chroms} ${metadata} ${log} --restore >"${out1}" 2>"${out2}"
  log_time "Training (restore) done"
  exit

else
  # --- kfold launch ---
  printf '%s\n' "python ${program_path}/epiatlas_training.py $category ${hparams} ${hdf5_list} ${chroms} ${metadata} ${log} > ${out1} 2> ${out2}"
  python ${program_path}/epiatlas_training.py $category ${hparams} ${hdf5_list} ${chroms} ${metadata} ${log} >"${out1}" 2>"${out2}"
fi
log_time "Training done"


# --- More logging ---
set +e

cd ${log}
merged_pred="full-10fold-validation_prediction.csv"
printf '\n%s\n' "Merging split*/validation_prediction.csv into ${merged_pred}"
# Keep one header at the top, then sort+uniq the data rows by ID.
split_files=( split*/validation_prediction.csv )
head -n 1 "${split_files[0]}" >"${merged_pred}"
tail -n +2 -q "${split_files[@]}" | sort -u >>"${merged_pred}"

to_augment="${log}/${merged_pred}"

printf '\n%s\n' "Launching following command"
printf '%s\n' "python ${program_path}/utils/augment_predict_file.py ${to_augment} ${metadata} --all-categories"
python ${program_path}/utils/augment_predict_file.py ${to_augment} ${metadata} --all-categories

printf '%s\n' "python ${program_path}/utils/create_confusion_matrices.py --from_prediction ${to_augment}"
python ${program_path}/utils/create_confusion_matrices.py --from_prediction ${to_augment}

# Copy slurm output file to log dir
if [[ -n "$SLURM_JOB_ID" ]]; then
  slurm_out_file="${SLURM_JOB_NAME}-*${SLURM_JOB_ID}.out"
  cp -v ${slurm_out_folder}/${slurm_out_file} ${log}/
fi
