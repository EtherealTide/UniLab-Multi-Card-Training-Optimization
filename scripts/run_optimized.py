"""Reproduce the tested GPU 0/1 SAC configuration on 5090-server2.

Only the MuJoCo pool is pinned. Do not wrap this command in taskset/numactl.
Training sizes and update frequencies remain the task defaults.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from experiment_paths import CPU_POOLS, runtime_environment, workspace_root

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=int, choices=(1, 2), default=2)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--single-pool", choices=("local32", "total64"), default="local32")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--workspace", type=Path, help="Directory containing the three source repositories; or UNILAB_WORKSPACE")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without running training")
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("iterations must be positive")
    root = workspace_root(args.workspace)
    log_dir = args.log_dir.resolve()
    if log_dir.exists():
        raise FileExistsError(f"Use a new experiment directory: {log_dir}")
    pools = CPU_POOLS
    command = [sys.executable, str(root / "UniLab/src/unilab/scripts/train_sac.py"),
        "task=g1_walk_flat/mujoco", "training.no_play=true",
        f"algo.max_iterations={args.iterations}", f"training.log_dir={log_dir}",
        f"training.trace_enabled={str(args.trace).lower()}"]
    if args.cards == 2:
        command += ["training.devices=[0,1]", "training.dp_collector_cpu_ids=" + json.dumps(pools, separators=(",", ":"))]
    else:
        cpus = pools[0] if args.single_pool == "local32" else pools[0] + pools[1]
        command += ["+env.cpu_ids=" + json.dumps(cpus, separators=(",", ":"))]
    env = runtime_environment(root) | {
        "CUDA_VISIBLE_DEVICES": "0,1", "NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "0"}
    for key in ("NCCL_GRAPH_MIXING_SUPPORT", "NCCL_GRAPH_STREAM_ORDERING", "NCCL_PROTO", "NCCL_ALGO"):
        env.pop(key, None)
    print(json.dumps({"command": command, "cards": args.cards, "collector_cpu_pools": pools,
        "note": "Host-specific preset; unrestricted learner affinity is required."}), flush=True)
    if args.dry_run:
        return
    raise SystemExit(subprocess.call(command, cwd=root / "UniLab", env=env))

if __name__ == "__main__":
    main()
