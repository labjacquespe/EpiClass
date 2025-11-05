#!/usr/bin/env python3
"""
Installation wrapper for epiclass.

Features:
- Auto-detects CPU/GPU
- Uses uv if available, otherwise pip
- Installs optional extras
- Exits if not in a virtual environment
"""
import os
import shutil
import subprocess
import sys

TORCH_VERSION = "2.6.0"


def main():
    """Main installation logic."""
    # ---------------------------
    # Ensure virtual environment
    # ---------------------------
    if sys.prefix == sys.base_prefix:
        print(
            "ERROR: You must activate a virtual environment before running this script."
        )
        sys.exit(1)

    # Change to the script's directory (where pyproject.toml is)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)

    # ---------------------------
    # Detect installer: uv or pip
    # ---------------------------
    def installer_cmd():
        """Return the installer command as a list, using uv if available, else pip."""
        if shutil.which("uv"):
            return ["uv", "pip", "install"]
        # else
        return [sys.executable, "-m", "pip", "install"]

    install_cmd = installer_cmd()

    # ---------------------------
    # Extras
    # ---------------------------
    extras = sys.argv[1:]  # e.g., test, dev

    # ---------------------------
    # Detect NVIDIA GPU / system
    # ---------------------------
    def is_computecanada():
        """Return True if running on Compute Canada system."""
        env_path = shutil.which("env")
        if not env_path:
            return False
        try:
            output = subprocess.check_output([env_path], universal_newlines=True)
            return "/cvmfs/soft.computecanada.ca/" in output
        except subprocess.SubprocessError:
            return False

    def has_nvidia_gpu():
        """Return True if an NVIDIA GPU is available via nvidia-smi."""
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            return False
        try:
            output = subprocess.check_output([nvidia_smi, "-L"], universal_newlines=True)
            return bool(output.strip())  # non-empty output => at least 1 GPU
        except subprocess.SubprocessError:
            return False

    # lazy evaluation, order matters
    if is_computecanada() or has_nvidia_gpu():
        target = "gpu"
    else:
        target = "cpu"

    # ---------------------------
    # Configure torch package and index
    # ---------------------------
    if target == "cpu":
        torch_pkg = f"torch=={TORCH_VERSION}"
        index_url = "https://download.pytorch.org/whl/cpu"
    else:
        torch_pkg = f"torch=={TORCH_VERSION}"
        index_url = None  # let pip pick wheels

    # ---------------------------
    # Install torch first
    # ---------------------------
    print(f"Installing {torch_pkg}...", flush=True)
    cmd = install_cmd + [torch_pkg]
    if index_url is not None:
        cmd += ["--index-url", index_url]
    subprocess.check_call(cmd)

    # ---------------------------
    # Install epiclass with extras
    # ---------------------------
    extras_str = ",".join(extras) if extras else None

    if extras_str is None:
        epiclass_spec = "."
    else:
        epiclass_spec = f".[{extras_str}]"

    print(
        f"Installing epiclass in editable mode with extra: '{extras_str}'...", flush=True
    )

    cmd = install_cmd + ["-e", epiclass_spec]
    print(f"Running command: {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)

    print("Installation complete.", flush=True)


if __name__ == "__main__":
    main()
