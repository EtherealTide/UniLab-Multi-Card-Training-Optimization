# 环境准备与诊断复现

以下命令假设已经按根 README 设置 `OPT_REPO` 和 `UNILAB_WORKSPACE`。所有训练使用 UniLab 的 Python 环境，从 `UniLab/` 执行 `uv run --no-sync`，不在本仓库另装一套 Torch。

## 新环境准备

在新的空工作区中克隆并固定版本；这些命令不是对已有脏仓库的重置指令：

```bash
mkdir -p "$UNILAB_WORKSPACE"
git clone https://github.com/unilabsim/UniLab.git "$UNILAB_WORKSPACE/UniLab"
git clone https://github.com/unilabsim/unilab_rl.git "$UNILAB_WORKSPACE/unilab_rl"
git clone https://github.com/unilabsim/unisim.git "$UNILAB_WORKSPACE/unisim"
git -C "$UNILAB_WORKSPACE/UniLab" checkout --detach 0fd2bd5d72210ad685838ade4ba4687b269f7de2
git -C "$UNILAB_WORKSPACE/unilab_rl" checkout --detach e2f18b4df1dcb0c35cd4c2dccd0517151633e53d
git -C "$UNILAB_WORKSPACE/unisim" checkout --detach 783220d64ba7a207d316f1dd06ccd1a1db1261bc
cd "$UNILAB_WORKSPACE/UniLab"
uv sync --extra mujoco
```

依赖安装、资产下载与运行条件以固定版本上游文档为准。这里没有重新打包 CUDA、NCCL、MuJoCo 或机器人资产。`uv sync` 完成不等于运行环境与历史实测完全相同；必须核对 Torch/CUDA/NCCL/driver，特别是 5090 的兼容 wheel、MuJoCo 3.11.0 和匹配的 mjbatch。实验不是对任意新版本性能的保证。

```bash
nvidia-smi
nvidia-smi topo -m
lscpu -e=CPU,CORE,SOCKET,NODE
uv run --no-sync python -c 'import torch,mujoco; print(torch.__version__,torch.version.cuda,torch.cuda.nccl.version(),mujoco.__version__); print(torch.cuda.can_device_access_peer(0,1))'
```

回到 README 应用补丁、设置 `PYTHONPATH`。若已应用补丁，`git apply --reverse --check PATCH` 应通过；无需再次应用。若使用不同上游版本，应先审查兼容性和重跑验证，而不是强行 `git apply --reject`。

## 指标口径与矩阵案例

`run_matrix.py` 保存每次命令、仓库 SHA、NCCL/可见 GPU 环境、tracked diff 快照、退出码、进程时间、TensorBoard 尾段指标。diff 快照不包含 untracked 新文件，最终补丁则包含新增测试。

| case | GPU | CPU 池 | 通信 |
|---|---|---|---|
| `n1` | 默认单卡0 | 上游默认 | 无DP |
| `n2_default` | 0,1 | 上游自动 | SHM禁用 |
| `n2_shm` | 0,1 | 上游自动 | SHM启用 |
| `n2_45` / `n2_04` | 4,5 / 0,4 | 上游自动 | SHM启用 |
| `n1_local32` | 单卡0 | 8–39 | 无DP |
| `n1_total64` | 单卡0 | 8–39,48–79 | 无DP |
| `n2_pool32` | 0,1 | 两个rank分别8–39、48–79 | SHM启用 |

主指标为最后50%事件中 `Δ累计总env steps / Δwall_time`。不是 `mean(Perf/total_fps)`，也不是 replay rows/s。诊断 trace、启动编译和正式无trace样本分别报告。所有批次保持每 rank 2048/8192/8/2；双卡全局 batch 和环境数加倍。

## CUDA Event 诊断

```bash
cd "$UNILAB_WORKSPACE/UniLab"
export PYTHONPATH="$UNILAB_WORKSPACE/UniLab/src:$UNILAB_WORKSPACE/unilab_rl/src:$UNILAB_WORKSPACE/unisim/src"
NCCL_SHM_DISABLE=0 uv run --no-sync python "$OPT_REPO/scripts/profile_training.py" \
  task=g1_walk_flat/mujoco 'training.devices=[0,1]' \
  training.no_play=true training.trace_enabled=true algo.max_iterations=1000 \
  "training.log_dir=$UNILAB_WORKSPACE/experiment_runs/events_diag"
```

该命令诊断自动CPU池；要诊断最终池配置，加 README 中完整的 `training.dp_collector_cpu_ids` 参数。`collective_events_rank*.json` 写在 rendezvous 所在的运行目录。external event 区间包含 pack/同步依赖/unpack，不是纯线缆传输时间。

## 通信与一致性

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_SHM_DISABLE=0 uv run --no-sync python \
  "$OPT_REPO/scripts/dp_probe.py" communication \
  --bytes 4 896976 3936444 --warmup 20 --iterations 100 --trials 3 \
  --output "$UNILAB_WORKSPACE/experiment_runs/comm_shm.json"

CUDA_VISIBLE_DEVICES=0,1 NCCL_SHM_DISABLE=0 uv run --no-sync python \
  "$OPT_REPO/scripts/dp_probe.py" correctness --compile-loss \
  --output "$UNILAB_WORKSPACE/experiment_runs/compiled_correctness.json"
```

约0.90/3.94MB的尺寸来自模型状态近似尺寸，不能称为精确的有效梯度payload；实际图路径还附带有限性标记。microbenchmark是隔离eager提交，不直接替代真实训练内的通信时间。

## Nsight Systems

先确认 `nsys` 可用，使用新的输出文件和运行目录：

```bash
NCCL_SHM_DISABLE=0 nsys profile \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --cuda-graph-trace=node -o "$UNILAB_WORKSPACE/experiment_runs/nsys_steady" \
  uv run --no-sync python "$OPT_REPO/scripts/nsys_training.py" \
  task=g1_walk_flat/mujoco 'training.devices=[0,1]' \
  training.no_play=true training.trace_enabled=false algo.max_iterations=210 \
  "training.log_dir=$UNILAB_WORKSPACE/experiment_runs/nsys_training"
```

脚本在约第190～200轮开启采样。逐节点追踪会明显扰动CPU提交和跨rank到达时差，不能用追踪后的耗时计算正式吞吐。原始 `.nsys-rep` 在证据包中；SQLite可用 `nsys export` 重新生成。

## 原补丁检查命令

```bash
cd "$UNILAB_WORKSPACE/unilab_rl"
CUDA_VISIBLE_DEVICES= PYTHONPATH="$PWD/src" uv run --no-sync pytest --cov=src/uni_rl -q
uv run --no-sync mypy src/uni_rl
uv run --no-sync pyright
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
```

该仓库不声称已重新跑过所有未来上游版本。历史结果与本次资料整理的自检分别保存，不把包装脚本的dry-run写成一次新的GPU性能测试。
