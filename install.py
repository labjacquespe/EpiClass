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


def argument_parser():
    """Return parsed command line arguments."""
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
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Increase output verbosity for installation commands."
    )
    # fmt: on
    return parser.parse_args()


def find_package_root(repo_root: Path) -> Path:
    """Find the package root by looking for pyproject.toml, starting from repo_root and searching recursively."""
    ignore = {".venv", ".git", "build", "dist", "__pycache__", "envs", "venv"}

    matches = []
    for path in repo_root.rglob("pyproject.toml"):
        if any(part in ignore for part in path.parts):
            continue
        matches.append(path)

    if not matches:
        raise RuntimeError(f"No pyproject.toml found under {repo_root}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple pyproject.toml files found: {matches}")

    return matches[0].parent


def installer_cmd(verbose: bool = False) -> list[str]:
    """Return the installer command as a list, using uv if available, else pip."""
    if shutil.which("uv"):
        base_cmd = ["uv", "pip", "install"]
        if verbose:
            return base_cmd + ["-v"]
        return base_cmd
    # else use pip
    base_cmd = [sys.executable, "-m", "pip", "install"]
    if verbose:
        return base_cmd + ["-vv"]
    return base_cmd


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
    # Change to package root, where pyproject.toml is.
    # ---------------------------
    REPO_ROOT = Path(__file__).resolve().parent
    PACKAGE_ROOT = find_package_root(REPO_ROOT)

    os.chdir(PACKAGE_ROOT)

    # ---------------------------
    # Read CLI arguments
    # ---------------------------
    cli = argument_parser()

    extras = cli.extras
    torch_version = cli.torch_version

    freeze_output = cli.freeze
    freeze_output = Path(freeze_output).resolve() if freeze_output else None

    # Detect installer: uv or pip
    install_cmd = installer_cmd(verbose=cli.verbose)

    # ---------------------------
    # Detect NVIDIA GPU / system
    # ---------------------------
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
    print(f"Installing {torch_pkg}... (${target})", flush=True)
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

        # Remove '-e' line
        lines = freeze_output.read_text(encoding="utf8").splitlines()
        lines = [line for line in lines if not line.startswith("-e ")]
        freeze_output.write_text("\n".join(lines) + "\n", encoding="utf8")

        print(f"Environment snapshot saved to {freeze_output}", flush=True)

    print("Installation complete.", flush=True)


if __name__ == "__main__":
    main()
