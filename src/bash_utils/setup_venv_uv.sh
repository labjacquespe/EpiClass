#!/bin/bash
# shellcheck disable=SC1091  # Don't warn about sourcing unreachable files

# Define your environment name and local module path.
# Default values
ENV_NAME="epiclass_env"
LOCAL_MODULE_PATH="src/python"

# Function to display the usage of the script
usage() {
  echo "Setup a Python virtual environment using 'uv' and install a local module."
  echo "Usage: $0 [-e virtual_env_name] [-s local_module_path]"
  echo "Defaults: env='$ENV_NAME', module='$LOCAL_MODULE_PATH'"
  exit 1
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -e|--env)
      ENV_NAME="$2"
      shift 2
      ;;
    -s|--source)
      LOCAL_MODULE_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown parameter passed: '$1'"
      echo "Use -h or --help for usage"
      exit 1
      ;;
  esac
done


# Check if local module path exists
if [ ! -d "$LOCAL_MODULE_PATH" ]; then
  echo "The specified local module path does not exist. This path must correspond to the package root. Please check the path: $LOCAL_MODULE_PATH"
  exit 1
else
  echo "Using package path: $LOCAL_MODULE_PATH"
fi


# Check if Python is installed
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python3 is not installed. Please install Python3 and rerun the script."
  exit 1
else
  echo "Python3 is installed: $(which python3)"
  echo "Python version: $(python3 --version)"
fi


# Check if uv is installed
if ! command -v uv >/dev/null 2>&1; then
  echo "'uv' is not installed. Please install it (try 'pipx install uv') and rerun the script."
  exit 1
fi


# Create a Python virtual environment
if [[ -n "$SLURM_JOB_ID" ]];
then
  echo "Running inside a SLURM job, creating the virtual environment in $SLURM_TMPDIR"
  cd "$SLURM_TMPDIR" || { echo "Failed to change directory to $SLURM_TMPDIR"; exit 1; }

  # extra precautions, so cvmfs python is used
  uv --no-managed-python --no-python-downloads venv ./$ENV_NAME
else
  echo "Creating the virtual environment in the current directory: $(pwd)"
  uv venv --seed ./$ENV_NAME
fi


# Activate the virtual environment
source "$ENV_NAME/bin/activate"

# Install local module
cd "$LOCAL_MODULE_PATH" || { echo "Failed to change directory to $LOCAL_MODULE_PATH"; exit 1; }
if ! uv pip install .; then # uses pyproject.toml
  echo "There was an error installing the local module."
  exit 1
fi

echo "Setup Complete"
