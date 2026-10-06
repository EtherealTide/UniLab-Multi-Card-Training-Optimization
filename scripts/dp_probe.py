"""Two-rank real NCCL probes; run from UniLab with its `uv run` environment.

Examples (physical placement is selected externally, never inferred):
  CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync "$OPT_REPO/scripts/dp_probe.py" correctness
  CUDA_VISIBLE_DEVICES=4,5 NCCL_SHM_DISABLE=0 uv run --no-sync "$OPT_REPO/scripts/dp_probe.py" communication
  CUDA_VISIBLE_DEVICES=0,4 NCCL_SHM_DISABLE=1 uv run --no-sync "$OPT_REPO/scripts/dp_probe.py" communication

Correctness defaults to eager loss kernels inside the PRODUCTION whole-cycle
CUDA graph. --compile-loss additionally exercises Inductor. This is a small
synthetic correctness test, NOT a training performance measurement. No setter
or synchronization implementation is patched: requires production DP support.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from uni_rl.ipc.dp_sync import DpParameterSync


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten(child, f"{prefix}/{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from flatten(child, f"{prefix}/{index}")
    else:
        yield prefix, value


def unchanged(before, after):
    left, right = dict(flatten(before)), dict(flatten(after))
    assert left.keys() == right.keys(), "State structure changed"
    changed = []
    for key in left:
        a, b = left[key], right[key]
        same = torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
        if not same:
            changed.append(key)
    assert not changed, f"Warmup changed state: {changed}"


def cross_rank_state(learner, device):
    """Compare all model buffers and every optimizer tensor against rank zero."""
    worst = 0.0
    tensor_count = 0
    for key, value in flatten(learner.get_state_dict()):
        if not isinstance(value, torch.Tensor):
            continue
        local = value.detach().to(device=device).contiguous()
        reference = local.clone()
        dist.broadcast(reference, src=0)
        assert torch.isfinite(local).all().item(), f"Nonfinite state at {key}"
        error = (local.double() - reference.double()).abs().max().item() if local.numel() else 0.0
        worst = max(worst, error)
        tensor_count += 1
    errors = torch.tensor([worst], device=device, dtype=torch.float64)
    dist.all_reduce(errors, op=dist.ReduceOp.MAX)
    # Identical averaged gradients and initial values must give identical states.
    assert errors.item() == 0, f"Rank state diverged: max_abs_error={errors.item()}"
    return {"max_abs_error": errors.item(), "state_tensors_checked": tensor_count}


def correctness(rank, args, sync, device):
    from uni_rl.algos.fast_sac.learner import FastSACLearner

    class EagerLossGraphLearner(FastSACLearner):
        def _compile_training_methods(self):
            # Keep the whole-cycle capture, optimizers and real NCCL unchanged.
            pass

    cls = FastSACLearner if args.compile_loss else EagerLossGraphLearner
    torch.manual_seed(3100 + rank)
    learner = cls(
        obs_dim=12, critic_obs_dim=15, action_dim=4, device=device,
        actor_hidden_dim=32, critic_hidden_dim=48, num_atoms=11,
        use_amp=False, obs_normalization=False,
    )
    try:
        return _correctness_checks(learner, rank, args, sync, device)
    finally:
        # Release captured NCCL graph nodes before worker closes the process
        # group, including failed assertions and compiled-function reference cycles.
        learner.set_gradient_sync(None)


def _correctness_checks(learner, rank, args, sync, device):
    from uni_rl.offpolicy.warmup import OffPolicyWarmupContext

    sync.broadcast_from_rank0(learner.dp_initial_sync_tensors())
    learner.set_gradient_sync(sync.allreduce_gradients)
    learner.set_gradient_graph_hooks(
        lambda: sync.begin_gradient_graph_capture(enable_timing=True),
        sync.end_gradient_graph_capture,
        sync.record_gradient_graph_replay,
    )
    assert learner.use_update_cycle, "Production whole-cycle graph must remain enabled"
    rows = args.batch_size * args.updates
    # Each rank deliberately sees different rewards, observations and actions.
    batch = {
        "obs": torch.randn(rows, 12, device=device) + rank,
        "critic": torch.randn(rows, 15, device=device) + rank,
        "actions": torch.randn(rows, 4, device=device).tanh(),
        "rewards": torch.randn(rows, device=device) + rank,
        "next_obs": torch.randn(rows, 12, device=device) + rank,
        "next_critic": torch.randn(rows, 15, device=device) + rank,
        "dones": torch.zeros(rows, device=device),
        "truncated": torch.zeros(rows, device=device),
    }
    batch["dones"][::7] = 1
    batch["truncated"][::14] = 1
    context = OffPolicyWarmupContext(
        inference_observations=batch["obs"][:2],
        inference_dones=batch["dones"][:2], batch_size=args.batch_size,
        updates_per_step=args.updates, policy_frequency=4, target_frequency=1,
        policy_before_critic=False, replay_batch=batch,
    )
    before = copy.deepcopy(learner.get_state_dict())
    before_cpu_rng = torch.random.get_rng_state().clone()
    before_cuda_rng = torch.cuda.get_rng_state(device).clone()
    dist.barrier()
    started = time.perf_counter()
    learner.prepare_for_collection(context)
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - started
    unchanged(before, learner.get_state_dict())
    assert torch.equal(before_cpu_rng, torch.random.get_rng_state()), "Warmup changed CPU RNG"
    assert torch.equal(before_cuda_rng, torch.cuda.get_rng_state(device)), "Warmup changed CUDA RNG"
    assert learner._update_cycle_graph is not None, "Graph was not captured"
    assert sync.take_gradient_sync_metrics() == (0.0, 0), "Warmup leaked sync metrics"
    checks = [cross_rank_state(learner, device)]
    snapshots = []
    replay_sync_metrics = []
    expected_calls = 2 * args.updates + (args.updates + 3) // 4
    for replay in range(args.replays):
        # Also test that the cached graph consumes new replay-buffer contents.
        batch["rewards"].add_(0.1 * (rank + 1))
        learner.update_cycle(
            batch, updates_per_step=args.updates, policy_frequency=4,
            target_frequency=1, policy_before_critic=False,
        )
        torch.cuda.synchronize()
        seconds, calls = sync.take_gradient_sync_metrics()
        assert calls == expected_calls, ("replay sync calls", replay, calls, expected_calls)
        assert seconds > 0, ("missing CUDA event timing", replay, seconds)
        replay_sync_metrics.append({"replay": replay, "sync_calls": calls, "cuda_event_sum_seconds": seconds})
        assert sync.take_gradient_sync_metrics() == (0.0, 0), "Metrics did not reset"
        checks.append(cross_rank_state(learner, device))
        snapshots.append(learner.log_alpha.item())
    final = learner.get_state_dict()
    updated = {}
    for name in ("actor", "qnet", "qnet_target", "log_alpha"):
        old = dict(flatten(before[name]))
        new = dict(flatten(final[name]))
        updated[name] = any(not torch.equal(old[k], new[k]) for k in old)
        assert updated[name], f"{name} never changed during graph replay"
    # Catch stale optimizer tensor addresses after warmup restore/capture.
    optimizer_steps = {}
    for name, updates_per_cycle in (
        ("q_optimizer", args.updates), ("alpha_optimizer", args.updates),
        ("actor_optimizer", (args.updates + 3) // 4),
    ):
        steps = [float(state["step"].item()) for state in final[name]["state"].values()]
        expected = args.replays * updates_per_cycle
        assert steps and all(step == expected for step in steps), (name, steps, expected)
        optimizer_steps[name] = {"min": min(steps), "max": max(steps), "expected": expected}

    # Keep the original replay/optimizer-step assertions above independent of
    # recapture and deliberately skipped updates below.
    new_updates = 4 if args.updates != 4 else 8
    new_batch_size = args.batch_size + 1
    indices = torch.arange(new_updates * new_batch_size, device=device) % rows
    new_batch = {key: value.index_select(0, indices) for key, value in batch.items()}
    recapture_before = copy.deepcopy(learner.get_state_dict())
    recapture_cpu_rng = torch.random.get_rng_state().clone()
    recapture_cuda_rng = torch.cuda.get_rng_state(device).clone()
    old_graph_key = learner._update_cycle_graph_cache_key
    learner.prepare_for_collection(OffPolicyWarmupContext(
        inference_observations=new_batch["obs"][:2],
        inference_dones=new_batch["dones"][:2], batch_size=new_batch_size,
        updates_per_step=new_updates, policy_frequency=4, target_frequency=1,
        policy_before_critic=False, replay_batch=new_batch,
    ))
    torch.cuda.synchronize()
    unchanged(recapture_before, learner.get_state_dict())
    assert torch.equal(recapture_cpu_rng, torch.random.get_rng_state()), "Recapture changed CPU RNG"
    assert torch.equal(recapture_cuda_rng, torch.cuda.get_rng_state(device)), "Recapture changed CUDA RNG"
    assert learner._update_cycle_graph is not None
    assert old_graph_key != learner._update_cycle_graph_cache_key, "Graph cache key did not change"
    assert sync.take_gradient_sync_metrics() == (0.0, 0), "Recapture warmup leaked sync metrics"
    learner.update_cycle(
        new_batch, updates_per_step=new_updates, policy_frequency=4,
        target_frequency=1, policy_before_critic=False,
    )
    torch.cuda.synchronize()
    recapture_seconds, recapture_calls = sync.take_gradient_sync_metrics()
    recapture_expected_calls = 2 * new_updates + (new_updates + 3) // 4
    assert recapture_calls == recapture_expected_calls, (recapture_calls, recapture_expected_calls)
    assert recapture_seconds > 0, "Recaptured graph did not produce CUDA event timing"
    recapture_check = cross_rank_state(learner, device)
    recapture_after = learner.get_state_dict()
    for name, delta in (
        ("q_optimizer", new_updates), ("alpha_optimizer", new_updates),
        ("actor_optimizer", (new_updates + 3) // 4),
    ):
        for key, state in recapture_after[name]["state"].items():
            previous_step = recapture_before[name]["state"][key]["step"].item()
            assert state["step"].item() == previous_step + delta, ("recapture optimizer step", name, key)

    # An infinite loss may have finite derivatives. Only rank zero reports an
    # invalid loss: the sentinel must nevertheless stop BOTH ranks' optimizers.
    assert any(not torch.equal(q, target) for q, target in zip(
        learner.qnet.parameters(), learner.qnet_target.parameters(), strict=True
    )), "Target must differ from critic for the skipped target-update test"
    invalid_loss_checks = []
    for name, parameters in (
        ("q_optimizer", list(learner.qnet.parameters())),
        ("alpha_optimizer", [learner.log_alpha]),
        ("actor_optimizer", list(learner.actor.parameters())),
    ):
        optimizer = getattr(learner, name)
        skip_before = copy.deepcopy(learner.get_state_dict())
        for parameter in parameters:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            parameter.grad.fill_(rank + 1.0)
        loss = torch.tensor(float("inf") if rank == 0 else 1.0, device=device)
        learner._sync_loss_gradients(parameters, loss)
        assert all(torch.isfinite(parameter.grad).all().item() for parameter in parameters)
        learner._arm_optimizer_finite_gate(optimizer, loss)
        assert learner._optimizer_found_inf.item() == 1.0, ("rank did not arm skip", rank, name)
        optimizer.step()
        if name == "q_optimizer":
            assert learner._q_update_finite.item() == 0.0, "Critic skip did not disable target update"
            learner.soft_update_target()
        torch.cuda.synchronize()
        unchanged(skip_before, learner.get_state_dict())
        skip_cross_rank = cross_rank_state(learner, device)
        _, skip_calls = sync.take_gradient_sync_metrics()
        assert skip_calls == 1, ("invalid-loss collective count", name, skip_calls)
        invalid_loss_checks.append({
            "optimizer": name, "bad_loss_rank": 0, "gradients_finite": True,
            "both_ranks_skipped": True, "model_optimizer_state_unchanged": True,
            "target_unchanged": True, "cross_rank": skip_cross_rank,
        })
    return {
        "passed": True, "compile_loss": args.compile_loss,
        "warmup_preserved_state_and_rng": True, "warmup_seconds": warmup_seconds,
        "cross_rank_checks": checks, "updated": updated,
        "optimizer_steps": optimizer_steps, "log_alpha_by_replay": snapshots,
        "replay_sync_metrics": replay_sync_metrics,
        "recapture": {
            "updates": new_updates, "batch_size": new_batch_size,
            "warmup_preserved_state_and_rng": True,
            "sync_calls": recapture_calls, "expected_sync_calls": recapture_expected_calls,
            "cuda_event_sum_seconds": recapture_seconds, "cross_rank": recapture_check,
        },
        "invalid_loss_checks": invalid_loss_checks,
        "scope": "synthetic correctness only; not a throughput baseline",
    }


def communication(rank, args, sync, device):
    results = []
    for nbytes in args.bytes:
        elements = max(1, nbytes // 4)
        data = torch.full((elements,), float(rank + 1), device=device)
        # Fixed gradients stay bounded, independent of repeated reductions.
        parameters = [torch.nn.Parameter(chunk.clone()) for chunk in data.tensor_split(16) if chunk.numel()]
        for parameter in parameters:
            parameter.grad = torch.full_like(parameter, float(rank + 1))
        for kind in ("flat_allreduce", "dp_pack_allreduce_unpack"):
            def operation():
                if kind == "flat_allreduce":
                    dist.all_reduce(data, op=dist.ReduceOp.SUM)
                    data.mul_(0.5)
                else:
                    sync.allreduce_gradients(parameters)

            for _ in range(args.warmup):
                operation()
            torch.cuda.synchronize()
            trials = []
            for _ in range(args.trials):
                dist.barrier()
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                wall = time.perf_counter()
                start.record()
                for _ in range(args.iterations):
                    operation()
                end.record()
                end.synchronize()
                trials.append({
                    "wall_us_per_call": (time.perf_counter() - wall) * 1e6 / args.iterations,
                    "cuda_us_per_call": start.elapsed_time(end) * 1000 / args.iterations,
                })
            tested = data if kind == "flat_allreduce" else torch.cat([p.grad for p in parameters])
            assert torch.allclose(tested, torch.full_like(tested, 1.5)), "Collective averaging failed"
            results.append({"bytes": elements * 4, "kind": kind, "trials": trials})
    return {"passed": True, "measurements": results,
            "timing_note": "CUDA events include NCCL stream dependencies; flat measurement includes averaging scale kernel. Wall timing includes final synchronization. No topology performance inference without repeated trials."}


def worker(rank, args, directory):
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    sync = DpParameterSync(world_size=2, rank=rank, rendezvous_path=f"{directory}/rendezvous", device=device, timeout_s=180)
    result = {"rank": rank, "device": torch.cuda.get_device_name(rank), "mode": args.mode}
    try:
        sync.start()
        result["environment"] = {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "NCCL_ALGO", "NCCL_PROTO")}
        result["torch"] = torch.__version__
        result["cuda"] = torch.version.cuda
        result["nccl"] = torch.cuda.nccl.version()
        result.update(correctness(rank, args, sync, device) if args.mode == "correctness" else communication(rank, args, sync, device))
    except Exception:
        result.update(passed=False, error=traceback.format_exc())
        raise
    finally:
        Path(directory, f"rank{rank}.json").write_text(json.dumps(result, indent=2))
        sync.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("correctness", "communication"))
    parser.add_argument("--output", type=Path, default=Path("dp_probe.json"))
    parser.add_argument("--compile-loss", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--replays", type=int, default=3)
    parser.add_argument("--bytes", type=int, nargs="+", default=[4, 262144, 1048576, 8388608, 16777216])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if torch.cuda.device_count() < 2:
        parser.error("Expose two GPUs with CUDA_VISIBLE_DEVICES")
    failure = None
    with tempfile.TemporaryDirectory(prefix="sac-dp-probe-") as directory:
        try:
            mp.spawn(worker, args=(args, directory), nprocs=2, join=True)
        except Exception as error:
            failure = str(error)
        ranks = [json.loads(path.read_text()) for path in sorted(Path(directory).glob("rank*.json"))]
    result = {"passed": failure is None and len(ranks) == 2 and all(r.get("passed") for r in ranks), "ranks": ranks, "spawn_error": failure}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
