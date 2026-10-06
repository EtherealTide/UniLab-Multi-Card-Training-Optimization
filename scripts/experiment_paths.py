"""Portable paths for a sibling UniLab / unilab_rl / unisim workspace."""
import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent
CPU_POOLS = [list(range(8, 40)), list(range(48, 80))]


def workspace_root(value=None):
    candidate = value or os.environ.get("UNILAB_WORKSPACE")
    if candidate is None:
        raise ValueError("Set UNILAB_WORKSPACE or pass --workspace: directory containing UniLab, unilab_rl and unisim")
    root = Path(candidate).expanduser().resolve()
    for repository in ("UniLab", "unilab_rl", "unisim"):
        if not (root / repository / "src").is_dir():
            raise FileNotFoundError(f"Missing source directory: {root / repository / 'src'}")
    return root


def output_root(value=None):
    return Path(value or os.environ.get("UNILAB_EXPERIMENT_DIR") or REPOSITORY / "runs").expanduser().resolve()


def runtime_environment(root):
    return os.environ | {
        "PYTHONPATH": os.pathsep.join(str(root / p / "src") for p in ("UniLab", "unilab_rl", "unisim")),
        "PYTHONUNBUFFERED": "1",
    }
