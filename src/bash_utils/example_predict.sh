#!/bin/bash
#SBATCH --time=:::time::: # ex: 6:00:00
#SBATCH --account=:::your-account:::
#SBATCH --job-name==:::your-job-name:::
#SBATCH --output=./slurm_files/%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1 #GPU not needed for inference
#SBATCH --mem=:::memory::: # ex: 16G
#SBATCH --mail-user=:::your-email:::
#SBATCH --mail-type=END,FAIL
# shellcheck disable=SC1091  # Don't warn about sourcing unreachable files

# -- NOTE: The values above in between ':::' are to be replaced by the user --

set -e # exit on error

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

gen_path="$HOME/:::project-path:::"
input_path="${gen_path}/epilap/input"
output_path="${gen_path}/epilap/output/logs"
gen_program_path="${gen_path}/sources/epiclass" # MODIFY: git root

for path in ${gen_path} ${gen_program_path} ${input_path} ${output_path}; do
  if [ ! -d ${path} ]; then
    echo "${path} is not a directory. Please check the path."
    exit 1
  else
    echo "Used directory: ${path}"
  fi
done


# --- choose category + hparams + source files ---
# MODIFY THINGS HERE

category="sex"
release="2022-epiatlas"
assembly="hg38"
basename="100kb_all_none"
list_name="${basename}-unknown-sex" # IMPORTANT

dataset=${assembly}"_"${release}  # ex: hg38_2018-10

export LAYER_SIZE="3000" # IMPORTANT
export NB_LAYER="1"

# IMPORTANT # IMPORTANT # IMPORTANT
base_log="${output_path}/${release}/${assembly}_${basename}_pearson/${category}_${NB_LAYER}l_${LAYER_SIZE}n"

model_dir="${base_log}/10fold/split0" # IMPORTANT
log="${base_log}/predict_unknown" # IMPORTANT

# last model checkpoint file
checkpoint_file=$(cat "${model_dir}/best_checkpoint.list" | tail -n1 | cut -f1 -d " ")


# --- Creating correct paths for epilap and launching ---

timestamp=$(date +%s)

hdf5_list="${input_path}/hdf5_list/${dataset}/${list_name}.list"
chromsizes="${input_path}/chromsizes/hg38.noy.chrom.sizes"
out1="${log}/output_job${SLURM_JOB_ID}_${SLURM_JOB_NAME}_${timestamp}.o"
out2="${log}/output_job${SLURM_JOB_ID}_${SLURM_JOB_NAME}_${timestamp}.e"

for path in ${hdf5_list} ${chromsizes} ${checkpoint_file}; do
  if [ ! -f ${path} ]; then
    echo "${path} is not a file. Please check the path."
    exit 1
  else
    echo "Input: ${path}"
  fi
done


# --- use correct environment ---

program_path="${gen_path}/sources/epiclass/src/python/epiclass"
cd ${program_path}

if [[ -n "$SLURM_JOB_ID" ]]; then
  # create venv on the fly
  cd $SLURM_TMPDIR
  python -m venv epiclass_env
  source epiclass_env/bin/activate
  python ${gen_program_path}/install.py &> job${SLURM_JOB_ID}_venv_setup.log
else
  source /path/to/preinstalled/venv/bin/activate # MODIFY
fi


# --- Pre-checks ---

printf '\n%s\n' "Launching following command"
printf '%s\n' "python ${program_path}/utils/check_dir.py ${log}"
python ${program_path}/utils/check_dir.py ${log}

printf '\n%s\n' "Launching following command"
printf '%s\n' "python ${program_path}/utils/check_dir.py --exists ${model_dir}"
python ${program_path}/utils/check_dir.py --exists ${model_dir}


# --- launch ---

# --model takes either a model directory (resolved via its best_checkpoint.list, as below)
# or a direct .ckpt file. Use the direct file (e.g. --model "${checkpoint_file}") to load a
# model off a mounted filesystem where the absolute paths inside best_checkpoint.list don't
# resolve. --outdir defaults to a 'predictions' dir next to the checkpoint when omitted.
printf '\n%s\n' "Launching following command"
printf '%s\n' "python ${program_path}/mains/predict.py --hdf5 ${hdf5_list} --model ${model_dir} --chromsize ${chromsizes} --outdir ${log} > ${out1} 2> ${out2}"
python ${program_path}/mains/predict.py --hdf5 ${hdf5_list} --model ${model_dir} --chromsize ${chromsizes} --outdir ${log} > ${out1} 2> ${out2}


# -- You could then augment the prediction file with new metadata if it is known --

# to_augment="${log}/test_prediction.csv"
# metadata="${input_path}/metadata/${dataset}_harmonizedv8.json"

# printf '\n%s\n' "Launching following command"
# printf '%s\n' "python ${program_path}/utils/augment_predict_file.py ${to_augment} ${metadata} --all-categories"
# python ${program_path}/utils/augment_predict_file.py ${to_augment} ${metadata} --all-categories
