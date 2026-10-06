"""Diagnostic harness: retain each captured collective's CUDA event duration.

Run with the same Hydra arguments as train_sac.py and trace_enabled=true.
Additional event reads make this a diagnostic run, not the headline benchmark.
"""
import json
from pathlib import Path
from experiment_paths import workspace_root

from uni_rl.ipc.dp_sync import DpParameterSync


original_take = DpParameterSync.take_gradient_sync_metrics
original_close = DpParameterSync.close
samples = {}


def take(self):
    result = original_take(self)
    if result[1] and self._gradient_graph_events:
        samples.setdefault(id(self), []).append(
            [start.elapsed_time(end) for start, end in self._gradient_graph_events]
        )
    return result


def close(self):
    records = samples.pop(id(self), None)
    if records:
        output = Path(self.rendezvous_path).parent / f"collective_events_rank{self.rank}.json"
        output.write_text(json.dumps({"rank": self.rank, "milliseconds_per_collective": records}))
    return original_close(self)


DpParameterSync.take_gradient_sync_metrics = take
DpParameterSync.close = close
import hydra
from unilab.scripts.train_offpolicy import main

root = workspace_root()
if __name__ == "__main__":
    hydra.main(
        version_base="1.3", config_path=str(root / "UniLab/src/unilab/conf/sac"), config_name="config"
    )(main)()
