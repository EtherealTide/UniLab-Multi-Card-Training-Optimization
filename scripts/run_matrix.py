"""Sequential real-training experiments, preserving every distinct run directory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from experiment_paths import CPU_POOLS, output_root, runtime_environment, workspace_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="+")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = workspace_root(args.workspace)
    output = output_root(args.output_dir)
    train = root / "UniLab/src/unilab/scripts/train_sac.py"
    base_env = runtime_environment(root) | {"NCCL_DEBUG": "WARN"}
    cases = {
        "n1": (None, {}),
        "n2_default": ([0, 1], {"NCCL_SHM_DISABLE": "1"}),
        "n2_shm": ([0, 1], {"NCCL_SHM_DISABLE": "0"}),
        "n2_45": ([4, 5], {"NCCL_SHM_DISABLE": "0"}),
        "n2_04": ([0, 4], {"NCCL_SHM_DISABLE": "0"}),
        "n1_local32": (None, {}),
        "n1_total64": (None, {}),
        "n2_pool32": ([0, 1], {"NCCL_SHM_DISABLE": "0"}),
    }
    invalid = set(args.cases) - cases.keys()
    if invalid:
        parser.error(f"Unknown cases: {sorted(invalid)}; available: {list(cases)}")
    if args.iterations <= 0 or args.repeats <= 0:
        parser.error("iterations and repeats must be positive")
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
    for case_index, case in enumerate(args.cases * args.repeats):
        devices, overrides = cases[case]
        name = f"{args.prefix}_{case}"
        if args.repeats > 1:
            name = f"{args.prefix}_r{case_index // len(args.cases) + 1}_{case}"
        run_dir = output / name
        if run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite experiment: {run_dir}")
        env = base_env | overrides
        command = [sys.executable, str(train), "task=g1_walk_flat/mujoco",
                   "training.no_play=true", f"algo.max_iterations={args.iterations}",
                   f"training.log_dir={run_dir}", f"training.trace_enabled={str(args.trace).lower()}"]
        if devices:
            command.append("training.devices=" + json.dumps(devices, separators=(",", ":")))
        if case == "n1_local32":
            command.append("+env.cpu_ids=" + json.dumps(CPU_POOLS[0], separators=(",", ":")))
        elif case == "n1_total64":
            command.append("+env.cpu_ids=" + json.dumps(CPU_POOLS[0] + CPU_POOLS[1], separators=(",", ":")))
        elif case == "n2_pool32":
            command.append("training.dp_collector_cpu_ids=" + json.dumps(CPU_POOLS, separators=(",", ":")))
        command.extend(args.override)
        if args.dry_run:
            print(json.dumps({"name": name, "command": command, "environment_overrides": overrides}))
            continue
        metadata = {"command": command, "environment_overrides": overrides,
                    "communication_environment": {key: value for key, value in env.items() if key.startswith("NCCL_") or key == "CUDA_VISIBLE_DEVICES"},
                    "revisions": {p: subprocess.check_output(["git", "-C", str(root / p), "rev-parse", "HEAD"], text=True).strip()
                                  for p in ("UniLab", "unilab_rl", "unisim")}}
        diff = subprocess.check_output(["git", "-C", str(root / "unilab_rl"), "diff"], text=True)
        metadata["rl_diff_sha256"] = hashlib.sha256(diff.encode()).hexdigest()
        (output / f"{name}.patch").write_text(diff)
        print(f"START {name}", flush=True)
        start = time.perf_counter()
        with (output / f"{name}.log").open("w") as log:
            result = subprocess.run(command, cwd=root / "UniLab", env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        metadata.update(exit_code=result.returncode, process_wall_seconds=time.perf_counter() - start)
        if result.returncode == 0:
            from analyze_run import analyze
            metrics = analyze(run_dir)
            (output / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
            fps = metrics["scalars"]["Perf/total_fps"]
            metadata["logged_tail_fps"] = fps["mean"]
            metadata["actual_tail_fps"] = (fps["last_step"] - fps["first_step"]) / fps["wall_seconds"]
            metadata["learner_tail_ms"] = metrics["scalars"]["Perf/learning_time"]["mean"] * 1000
        (output / f"{name}_execution.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
