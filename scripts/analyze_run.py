"""Summarize completed real-training runs without rewriting their artifacts."""
import argparse
import collections
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def analyze(path):
    path = Path(path)
    summary = json.loads((path / "run_summary.json").read_text())
    config = json.loads((path / "run_config.json").read_text())
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalars = {}
    for tag in accumulator.Tags()["scalars"]:
        series = accumulator.Scalars(tag)
        tail = series[len(series) // 2:]
        values = sorted(e.value for e in tail)
        scalars[tag] = {
            "n": len(tail), "mean": sum(values) / len(values),
            "median": values[len(values) // 2],
            "p90": values[min(len(values) - 1, int(len(values) * .9))],
            "last": tail[-1].value,
            "first_step": tail[0].step, "last_step": tail[-1].step,
            "wall_seconds": tail[-1].wall_time - tail[0].wall_time,
        }
    trace = path / "perfetto_offpolicy_timeline.json"
    slices = collections.defaultdict(list)
    if trace.exists():
        data = json.loads(trace.read_text())
        events = data.get("traceEvents", []) if isinstance(data, dict) else data
        ends = [e["ts"] for e in events if e.get("name") == "learner/update_phase"]
        cutoff = sorted(ends)[len(ends) // 2] if ends else 0
        for e in events:
            if e.get("ph") == "X" and e.get("ts", 0) >= cutoff:
                slices[e["name"]].append(e["dur"] / 1000)
    phases = {key: {"n": len(vals), "mean_ms": sum(vals)/len(vals)}
              for key, vals in slices.items()}
    return {"path": str(path), "summary": summary, "config": config,
            "scalars": scalars, "trace_phases": phases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.run)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({"summary": result["summary"],
                      "perf": {k: v for k, v in result["scalars"].items()
                               if k.startswith("Perf/")},
                      "phases": result["trace_phases"]}, indent=2))
