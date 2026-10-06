"""Collect ten steady-state iterations with Nsight; diagnostic only."""
from experiment_paths import workspace_root

import hydra
import torch
from uni_rl.ipc.dp_sync import DpParameterSync
from unilab.scripts.train_offpolicy import main

original_take = DpParameterSync.take_gradient_sync_metrics
original_statistics = DpParameterSync.allreduce_statistics
counts = {}

def take(self):
    result = original_take(self)
    count = counts.get(id(self), 0) + 1
    counts[id(self)] = count
    if count == 190:
        torch.cuda.profiler.start()
    elif count == 200:
        torch.cuda.profiler.stop()
    return result

def statistics(self, **kwargs):
    with torch.cuda.nvtx.range("dp/statistics"):
        return original_statistics(self, **kwargs)

DpParameterSync.take_gradient_sync_metrics = take
DpParameterSync.allreduce_statistics = statistics
root = workspace_root()
if __name__ == "__main__":
    hydra.main(version_base="1.3", config_path=str(root / "UniLab/src/unilab/conf/sac"),
               config_name="config")(main)()
