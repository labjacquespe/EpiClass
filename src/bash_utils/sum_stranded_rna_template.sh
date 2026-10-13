#!/bin/bash
#SBATCH --time=:::time::: # ex: 1:00:00 for ~1500 pairs at 1kb
#SBATCH --account=:::your-account:::
#SBATCH --job-name=:::your-job-name:::
#SBATCH --output=:::slurm-out-folder:::/%x-job%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=:::memory::: # ex: 4G
#SBATCH --mail-user=:::your-email:::
#SBATCH --mail-type=END,FAIL
# shellcheck disable=SC1091  # Don't warn about sourcing unreachable files

# -- NOTE: The values above in between ':::' are to be replaced by the user. --
#
# Sum stranded RNA-Seq HDF5 pairs (plusRaw + minusRaw) into unstranded
# single-sample HDF5 files with utils/preprocessing/sum_stranded_rna_hdf5.py.
#
# Inputs are staged to node-local $SLURM_TMPDIR, summed there, and the results
# copied to the output dir at the end. Only the files named in the pair list are
# staged. The pair list format is described in the python script's docstring
# (2 columns: path_a path_b, or 3: new_id path_a path_b; comma/whitespace
# separated). Only the *basenames* of its paths matter here: --hdf5-dir points
# the python script at the staged copies.
#
# Usage:
#   sbatch <script>                         # full run
#   sbatch <script> --overwrite             # full run, replace existing outputs
#   sbatch --time=0:20:00 <script> --test   # first pair only, into a fresh test dir
#
# --test: only the first pair is staged and summed, and the results go to
#   ${work_dir}/test_job<jobid>_<date>/, never to the real output dir.
# --overwrite: without it, a full run refuses to start if the output dir already
#   holds results. The check runs before any staging, so a mistake costs nothing.
#   (The python script's own "skip existing outputs" does not protect anything
#   here: it writes to a fresh node-local dir on every run.)

set -e # exit on error

log_time() {
  echo "[$(date +%F_%T)] $1"
}

test_mode=false
overwrite=false
for arg in "$@"; do
  case "${arg}" in
    --test) test_mode=true ;;
    --overwrite) overwrite=true ;;
    *) echo "Unknown argument: ${arg}" >&2; exit 1 ;;
  esac
done

export PYTHONUNBUFFERED=TRUE

# For HPC environments
module purge
ml StdEnv/2023 python/3.11

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "=========================================="
  echo "SLURM_JOB_ID = $SLURM_JOB_ID"
  echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
  echo "SLURM_TMPDIR = $SLURM_TMPDIR"
  echo "TEST_MODE = ${test_mode}"
  echo "OVERWRITE = ${overwrite}"
  echo "=========================================="
else
  echo "This script stages files to \$SLURM_TMPDIR and must run as a SLURM job, exiting." >&2
  exit 1
fi


# --- paths ---
# MODIFY THINGS HERE

gen_program_path=":::path-to-epiclass-git-root:::"
python_script="${gen_program_path}/src/python/epiclass/utils/preprocessing/sum_stranded_rna_hdf5.py"

work_dir=":::work-dir:::"        # holds the pair list; gets the SLURM log and test runs
output_dir=":::output-dir:::"    # final summed HDF5s + pair_mapping.tsv
pair_list="${work_dir}/:::pair-list-name:::"  # ex: rna_pairs.csv

slurm_out_folder=":::slurm-out-folder:::"  # same as in the #SBATCH --output line

# Where the stranded input HDF5s are read from. Pick one method:
#   "dir": a plain directory of HDF5 files (copied with rsync).
#   "dar": a dar archive (extracted with dar; only the needed members).
staging_method="dir" # MODIFY: "dir" or "dar"
input_hdf5_dir=":::input-hdf5-dir:::"      # "dir" method
dar_archive=":::dar-archive-basename:::"   # "dar" method: path without the .N.dar suffix
dar_member_prefix=":::dar-member-prefix:::" # "dar" method: path of the HDF5s inside the
                                            # archive, ex: 1kb_all_none/1kb_all_none/


for path in ${gen_program_path} ${work_dir} ${output_dir} ${slurm_out_folder}; do
  if [ ! -d ${path} ]; then
    echo "${path} is not a directory. Please check the path." >&2
    exit 1
  else
    echo "Used directory: ${path}"
  fi
done
if [ ! -f ${pair_list} ]; then
  echo "Pair list ${pair_list} does not exist." >&2
  exit 1
fi

# Record which version of the code is being run.
cd ${gen_program_path} || { echo "Could not cd into '${gen_program_path}', exiting"; exit 1; }
echo "GIT COMMIT: $(git describe --always)"
cd - &> /dev/null


# --- refuse to replace existing results unless asked (test runs use a fresh dir) ---

if [[ "${test_mode}" == false ]]; then
  n_existing=$(find ${output_dir} -maxdepth 1 \( -name '*.hdf5' -o -name 'pair_mapping.tsv' \) | wc -l)
  if [[ ${n_existing} -gt 0 ]]; then
    if [[ "${overwrite}" == true ]]; then
      echo "WARNING: --overwrite set, ${n_existing} existing file(s) in ${output_dir} will be replaced."
    else
      echo "${output_dir} already holds ${n_existing} result file(s) (*.hdf5 / pair_mapping.tsv)." >&2
      echo "Re-submit with --overwrite to replace them, or move them away first. Exiting." >&2
      exit 1
    fi
  fi
fi


# --- where results and logs go (test runs never touch the real output dir) ---
# Logs must land on persistent storage: $SLURM_TMPDIR is wiped when the job ends.

if [[ "${test_mode}" == true ]]; then
  test_output_dir="${work_dir}/test_job${SLURM_JOB_ID}_$(date +%F_%H-%M-%S)"
  mkdir -p ${test_output_dir}
  final_dir=${test_output_dir}
  final_log_dir=${test_output_dir}
else
  final_dir=${output_dir}
  final_log_dir=${work_dir}
fi


# --- use correct environment ---

venv_log="${final_log_dir}/job${SLURM_JOB_ID}_venv_setup.log"
log_time "Starting venv setup (log: ${venv_log})"
cd $SLURM_TMPDIR
python -m venv epiclass_env
source epiclass_env/bin/activate
python ${gen_program_path}/install.py &> ${venv_log}
cd - &> /dev/null
log_time "Venv setup done"


# --- pairs to process + input files they need ---

if [[ "${test_mode}" == true ]]; then
  # First pair only.
  test_pair_list="$SLURM_TMPDIR/test_pair_list.txt"
  grep -v -m 1 -E '^[[:space:]]*(#|$)' ${pair_list} > ${test_pair_list}
  pair_list=${test_pair_list}
  echo "Test pair:"
  cat ${pair_list}
fi

# Basenames of the two input files of each pair (the last two fields of a line).
needed_files="$SLURM_TMPDIR/needed_files.txt"
awk -F'[[:space:],]+' '
  /^[[:space:]]*(#|$)/ { next }
  { sub(/[[:space:],]+$/, "") }
  { n = split($(NF-1), a, "/"); print a[n]; n = split($NF, b, "/"); print b[n] }
' ${pair_list} | sort -u > ${needed_files}
n_needed=$(wc -l < ${needed_files})
echo "Input files needed: ${n_needed}"


# --- stage input files to node scratch ---

tmp_hdf5_dir="$SLURM_TMPDIR/stranded_hdf5"
tmp_output_dir="$SLURM_TMPDIR/unstranded_hdf5"
mkdir -p ${tmp_hdf5_dir} ${tmp_output_dir}

log_time "Staging inputs (${staging_method}) to ${tmp_hdf5_dir}"
case "${staging_method}" in
  dir)
    # Missing files make rsync exit non-zero; the count check below reports them.
    rsync -a --files-from=${needed_files} ${input_hdf5_dir}/ ${tmp_hdf5_dir}/ || true
    ;;
  dar)
    dar_include="$SLURM_TMPDIR/dar_include.txt"
    sed "s|^|${dar_member_prefix}|" ${needed_files} > ${dar_include}
    # -f (flat): restore without the archive's directory structure.
    dar -Q -O -w -f --include-from-file=${dar_include} -x ${dar_archive} -R ${tmp_hdf5_dir}
    ;;
  *)
    echo "Unknown staging_method '${staging_method}'" >&2
    exit 1
    ;;
esac
log_time "Staging done"

n_staged=$(find ${tmp_hdf5_dir} -maxdepth 1 -name '*.hdf5' | wc -l)
if [[ ${n_staged} -ne ${n_needed} ]]; then
  echo "Staged ${n_staged} HDF5 file(s) but the pair list needs ${n_needed}." >&2
  echo "Missing:" >&2
  comm -23 ${needed_files} <(find ${tmp_hdf5_dir} -maxdepth 1 -type f -printf '%f\n' | sort) >&2
  exit 1
fi


# --- launch ---

log_time "Summing pairs"
python ${python_script} \
  --pair-list ${pair_list} \
  --hdf5-dir ${tmp_hdf5_dir} \
  --output-dir ${tmp_output_dir} \
  --per-pair
log_time "Summing done"


# --- copy results from node scratch ---

# "src/." copies the directory's *contents*. "cp -r src/ dst" with an existing dst
# would nest them as dst/unstranded_hdf5/... (the trailing slash does not prevent it).
log_time "Copying results to ${final_dir}"
cp -r ${tmp_output_dir}/. ${final_dir}/
log_time "Copy done"

n_written=$(find ${tmp_output_dir} -maxdepth 1 -name '*.hdf5' | wc -l)
n_final=$(find ${final_dir} -maxdepth 1 -name '*.hdf5' | wc -l)
echo "HDF5 files written on node: ${n_written}, now in ${final_dir}: ${n_final}"

if [[ "${test_mode}" == true ]]; then
  ls -l ${final_dir}
  cat ${final_dir}/pair_mapping.tsv
fi


# --- more logging ---
set +e

slurm_out_file="${SLURM_JOB_NAME}-job${SLURM_JOB_ID}.out"
cp -v ${slurm_out_folder}/${slurm_out_file} ${final_log_dir}/
