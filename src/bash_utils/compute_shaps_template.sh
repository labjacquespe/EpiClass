#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --account=:::account:::
#SBATCH --job-name=:::job_name:::
#SBATCH --output=/lustre06/project/6007017/rabyj/epilap/output/sub/slurm_files/%x-job%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=80G
#SBATCH --mail-user=:::email:::
#SBATCH --mail-type=END,FAIL
# shellcheck disable=SC2317  # Don't warn about unreachable commands in this file
# shellcheck disable=SC2086  # Don't warn about double quoting
# shellcheck disable=SC1091  # Don't warn about sourcing unreachable files

log_time() {
    echo "Time: $(date +%F_%T) - $1"
}

log_time "Starting script"

ml StdEnv/2020 python/3.8

export PYTHONUNBUFFERED=TRUE

if [[ -n "$SLURM_JOB_ID" ]];
then
  echo "print =========================================="
  echo "print SLURM_JOB_ID = $SLURM_JOB_ID"
  echo "print SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
  echo "print =========================================="
fi

gen_path="/lustre06/project/6007017/rabyj"
input_path="${gen_path}/epilap/input"
gen_program_path="${gen_path}/sources/epiclass"
program_path="${gen_program_path}/src/python/epiclass"

slurm_out_folder="${gen_path}/epilap/output/sub/slurm_files"

# use correct environment
log_time "Loading and/or creating the virtual environment"
set -e
if [[ -n "$SLURM_JOB_ID" ]];
then
  cd $SLURM_TMPDIR
  bash ${gen_program_path}/src/bash_utils/setup_venv.sh -r ${gen_program_path}/requirements/minimal_requirements.txt -s ${gen_program_path}/src/python &> ${slurm_out_folder}/${SLURM_JOB_ID}_setup.log
  source epiclass_env/bin/activate
else
  source /lustre07/scratch/rabyj/envs/epiclass/bin/activate
fi


# --- choose category + hparams + source files ---

model="NN" # IMPORTANT

category=:::category::: # IMPORTANT

# -- NN --

# IMPORTANT!!!!
model_path=":::model_path:::"
output_log=":::output_log:::"

# --- Creating correct paths for programs/launching ---

timestamp=$(date +%s)

list_background=":::background_list:::"
list_explain=":::eval_list:::"

chroms="${input_path}/chromsizes/hg38.noy.chrom.sizes"
out1="${output_log}/output_job${SLURM_JOB_ID}_${SLURM_JOB_NAME}_${timestamp}.o"
out2="${output_log}/output_job${SLURM_JOB_ID}_${SLURM_JOB_NAME}_${timestamp}.e"

log_time "Evaluating paths."
for path in ${list_background} ${list_explain} ${chroms}; do
  if [ ! -f ${path} ]; then
    echo "${path} is not a file. Please check the path."
    exit 1
  else
    echo "Input: ${path}"
  fi
done

# --- Pre-checks ---
log_time "Executing pre-checks"
set -e # in case check_dir fails, to stop bash script
program_path="${gen_path}/sources/epiclass/src/python/epiclass"
cd ${program_path}

printf '\n%s\n' "Launching following command"
printf '%s\n' "python ${program_path}/utils/check_dir.py --exists ${model_path}"
python ${program_path}/utils/check_dir.py --exists ${model_path}

printf '%s\n' "python ${program_path}/utils/check_dir.py ${output_log}"
python ${program_path}/utils/check_dir.py ${output_log}

# Preconditions passed, copy launch script to log dir.
if [[ -n "$SLURM_JOB_ID" ]];
then
scontrol write batch_script ${SLURM_JOB_ID} ${output_log}/launch_script_${SLURM_JOB_NAME}-job${SLURM_JOB_ID}.sh
fi


# --- Transfer files to node scratch ---

if [[ -n "$SLURM_JOB_ID" ]];
then
  log_time "Extracting hdf5s"
  tar_file=":::full_tar_path:::"  # IMPORTANT

  cd $SLURM_TMPDIR

  echo "Untaring $tar_file in $SLURM_TMPDIR"
  tar -xf $tar_file

  # Finding correct hdf5 folder name
  folder_name=$(find . -mindepth 1 -type d | grep -v "epiclass" | head -n 1)
  folder_name=$(basename $folder_name)
  echo "Using folder_name: $folder_name"
  export HDF5_PARENT="${folder_name}" # IMPORTANT
  log_time "Extraction done"
fi


# --- MAIN PROGRAM ---
set +e

# usage: compute_shaps.py [-h] -m {NN,LGBM} --background_hdf5 background-hdf5 --explain_hdf5 explain-hdf5 --chromsize CHROMSIZE [-l LOGDIR] [-o --output-name]
#                        [--model_file model_file] [--model_dir MODEL_DIR]

basecmd="python ${program_path}/compute_shaps.py -m ${model} --background_hdf5 ${list_background} --explain_hdf5 ${list_explain} --chromsize ${chroms} -l ${output_log} -o explain_${category}"
basecmd="${basecmd} --model_dir ${model_path}"

log_time "Running main program"
printf '\n%s\n' "Launching following command"
printf '%s\n' "${basecmd} > ${out1} 2> ${out2}"
${basecmd} > ${out1} 2> ${out2}
log_time "Main program done"


# Copy slurm output file to log dir
if [[ -n "$SLURM_JOB_ID" ]];
then
  slurm_out_file="${SLURM_JOB_NAME}-*${SLURM_JOB_ID}.out"
  cp -v ${slurm_out_folder}/${slurm_out_file} ${output_log}/
fi

log_time "END OF SCRIPT"
