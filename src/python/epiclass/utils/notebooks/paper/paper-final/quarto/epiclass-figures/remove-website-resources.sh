#!/bin/bash

# Remove unnecessary website resources
# We can do this because we used the embed-resources option in quarto.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
folder_name=$(basename "$SCRIPT_DIR")

if [ -z "$SCRIPT_DIR" ]; then
    echo "Could not determine script directory. Exiting."
    exit 1
elif [ "$folder_name" != "epiclass-figures" ]; then
    echo "Script directory is incorrect: $SCRIPT_DIR"
    exit 1
fi

# path found using realpath -s --relative-to="$this_folder" "$target_folder"
docs_path=${SCRIPT_DIR}/../../../../../../../../../docs/epiclass-paper

rm -r ${docs_path}/resources
rm ${docs_path}/figs/*.qmd
