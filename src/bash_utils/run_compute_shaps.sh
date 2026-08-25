#!/bin/bash
base_dir="$HOME/Projects"

code="$base_dir/sources/epiclass/src/python/epiclass/mains/compute_shaps.py"
env="$base_dir/epilap/venv-epilap-pytorch/bin/activate"

. $env

input="$base_dir/epilap/input"
hdf5_list="$input/hdf5_list/hg38_2022-epiatlas/estimator-debug-biotype-n400.list"
metadata="$input//metadata/merge_EpiAtlas_allmetadata-v11-mod.json"
chroms="$input/chromsizes/hg38.noy.chrom.sizes"

output="$base_dir/epilap/output"
logdir="$output/logs/hg38_2022-epiatlas/shap"
model_dir="$output/models/split0"

# compute_shaps.py category hdf5 chromsize metadata logdir model
python $code "assay" $hdf5_list $chroms $metadata $logdir $model_dir
