#!/usr/bin/env python3
"""
Installation wrapper for epiclass.

Features:
- Auto-detects CPU/GPU
- Uses uv if available, otherwise pip
- Installs optional extras
- Exits if not in a virtual environment
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_TORCH = "2.6.0"


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
    # Argument parsing
    # ---------------------------
    # fmt: off
    parser = argparse.ArgumentParser(description="Install epiclass and dependencies.")
    parser.add_argument(
        "extras",
        nargs="*",
        help="Optional extras to install (e.g. dev, test)",
    )
    parser.add_argument(
        "--freeze",
        nargs="?",
        const="installed-packages.txt",
        metavar="FILE",
        help="Write pip freeze output to FILE (default: installed-packages.txt)",
    )
    parser.add_argument(
        "--torch-version",
        type=str,
        default=DEFAULT_TORCH,
        help=f"Specify the PyTorch version to install (default: {DEFAULT_TORCH})",
    )
    # fmt: on

    args = parser.parse_args()
    extras = args.extras

    freeze_output = args.freeze
    freeze_output = Path(freeze_output).resolve() if freeze_output else None

    torch_version = args.torch_version

    # Change to the script's directory (where pyproject.toml is)
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

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
        torch_pkg = f"torch=={torch_version}"
        index_url = "https://download.pytorch.org/whl/cpu"
    else:
        torch_pkg = f"torch=={torch_version}"
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

    # ---------------------------
    # Optional freeze
    # ---------------------------
    if freeze_output:
        print(f"Freezing environment package list to {freeze_output}...", flush=True)
        Path(freeze_output).parent.mkdir(parents=True, exist_ok=True)

        if shutil.which("uv"):
            freeze_cmd = ["uv", "pip", "freeze"]
        else:
            freeze_cmd = [sys.executable, "-m", "pip", "freeze"]

        with open(freeze_output, "w", encoding="utf8") as f:
            subprocess.check_call(freeze_cmd, stdout=f)

        print(f"Environment snapshot saved to {freeze_output}", flush=True)

    print("Installation complete.", flush=True)


if __name__ == "__main__":
    main()
