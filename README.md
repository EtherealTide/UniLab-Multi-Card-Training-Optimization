# UniLab 双卡 SAC 训练优化

在真实 **UniLab / MuJoCo / G1WalkFlat / SAC** 训练中，RTX 5090 单卡平均 **50,194**、双卡 **108,646 env steps/s**，均值比 **2.165×**。保留 Inductor + 整轮 CUDA Graph，每 rank 使用 2048 环境、8192 batch、每轮 8 次 critic / temperature 更新与 2 次 actor 更新。

| 配置（各 3 次） | 单卡 steps/s | 双卡 steps/s | 均值比 |
|---|---:|---:|---:|
| 图内梯度同步 + SHM，自动 CPU 配置 | 30,766 | 52,750 | 1.715× |
| 仿真池共 64 核：单池 64 / 双池各 32 | 45,510 | 108,646 | 2.387× |
| **本地仿真池：单卡 32 / 双卡各 32** | **50,194** | **108,646** | **2.165×** |

结论：图内 all-reduce 可以与整轮 CUDA Graph 共存；实际瓶颈还包括 CPU 仿真、IPC 和跨 rank 等待。推荐 GPU 0–1、NCCL SHM，以及仅设置仿真线程池 CPU 列表，保持 learner 原有亲和性。原始默认双卡因代码保护直接报错，没有有效基线 FPS。

这是 CPU 配额、collector 和 GPU 一起扩展的**端到端弱扩展**，不是补丁独立收益或纯 GPU 扩展效率。双卡全局环境数 / batch 加倍；尚未验证长期收敛。全部版本、参数、逐次数据、分析与限制见 [技术报告](docs/REPORT_zh.md)。

## 直接在 UniLab 中使用

以下命令在已有 UniLab 环境中运行；示例路径替换成你自己的路径。**前提是当前 Python 实际导入的 `uni_rl` 已包含本文优化**，仅更新 UniLab 或本 README 不会更新其已安装依赖，`--no-sync` 也不会安装新版本。只有满足该前提，才可以仅使用 UniLab 项目直接启动优化后的双卡训练。

先在 UniLab 根目录检查实际加载位置和图同步接口：

```bash
uvx uv@0.12.5 run --no-sync python -c 'import inspect; import uni_rl.algos.fast_sac.learner as m; print(m.__file__); print("graph hooks:", any(hasattr(c, "set_gradient_graph_hooks") for _, c in inspect.getmembers(m, inspect.isclass)))'
```

当前补丁的接口检查应为 `graph hooks: True`（这是识别本次实现的检查，不是所有未来版本的兼容性测试）。若报 `FastSAC NVIDIA CUDA whole-cycle mode does not support DP fallback`，实际加载的仍是带旧限制的代码；换 GPU、CPU 池或 uv 版本不能解决。这里不预设该优化已经发布到某个安装包版本。

**已有优化源码的实验工作区**：若检查指向 `.venv/.../site-packages/uni_rl`，但修改后的源码在相邻 `unilab_rl/`，可在当前终端选择该源码，再重复检查及下方命令，无需重新应用补丁：

```bash
# 仅适用于现有 UniLabSim/UniLab 与 UniLabSim/unilab_rl 布局。
export PYTHONPATH="$(realpath ../unilab_rl/src)${PYTHONPATH:+:$PYTHONPATH}"
```

如果只有 UniLab 且安装的 RL 依赖尚未包含优化，则还不满足直接启动这条优化路径的条件。

### 单卡与双卡对照（UniLab 自带 benchmark）

```bash
cd /mnt/gfs/home/liangyu/desktop/UniLabSim/UniLab
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=0 \
uvx uv@0.12.5 run --no-sync \
  scripts/benchmark/rl/benchmark_offpolicy_dp_scaling.py \
  --iterations 60 \
  --devices 0,1 \
  --extra-overrides \
    training.trace_enabled=true
```

该命令**先跑单卡，再跑双卡**，任务默认是 SAC / G1WalkFlat / MuJoCo。选择物理 GPU 4、5 时，把 `CUDA_VISIBLE_DEVICES=0,1` 改为 `CUDA_VISIBLE_DEVICES=4,5`，`--devices 0,1` 保持不变，因为它使用可见设备的逻辑索引。

仅运行单卡 benchmark（例如物理 GPU 4）：

```bash
CUDA_VISIBLE_DEVICES=4 uvx uv@0.12.5 run --no-sync \
  scripts/benchmark/rl/benchmark_offpolicy_dp_scaling.py \
  --iterations 60 \
  --devices 0 \
  --extra-overrides \
    training.trace_enabled=true
```

当前脚本中，`--devices` 只有一个索引时仅跳过双卡组，单卡基线仍使用默认可见 GPU；所以单卡选卡应通过 `CUDA_VISIBLE_DEVICES` 完成。`--extra-overrides` 中的配置会同时传给单卡和双卡组。

结果默认写入 `scripts/benchmark/outputs/offpolicy_dp_scaling/results.json`，训练日志在其 `runs/` 下。重复运行会替换同名 `n1/n2` 目录；需要保留时先复制旧结果，单独修改 `--out-json` 不会更改训练目录。

60 轮 + trace 适合快速诊断；正式测量使用 `--iterations 1000` 和 `training.trace_enabled=false`。此 benchmark 的 Steps/s 是尾段 `Perf/total_fps` 均值，与本报告的累计环境步数差 / 墙钟时间差不同，不能直接拿其 scaling 数字与 2.165× 比较。

### 只启动单卡或双卡训练

如果不需要先跑单卡基线，直接使用 UniLab 训练入口。以下两个命令任选其一；无需额外调用 `torchrun`：

```bash
cd /mnt/gfs/home/liangyu/desktop/UniLabSim/UniLab

# 只跑单卡，物理 GPU 4。
CUDA_VISIBLE_DEVICES=4 uvx uv@0.12.5 run --no-sync \
  src/unilab/scripts/train_sac.py \
  task=g1_walk_flat/mujoco 'training.devices=[0]' \
  algo.max_iterations=1000 training.no_play=true training.trace_enabled=false

# 只跑双卡，物理 GPU 0、1；换成4、5只需修改CUDA_VISIBLE_DEVICES。
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=0 \
uvx uv@0.12.5 run --no-sync \
  src/unilab/scripts/train_sac.py \
  task=g1_walk_flat/mujoco 'training.devices=[0,1]' \
  algo.max_iterations=1000 training.no_play=true training.trace_enabled=false
```

日志使用 UniLab 默认的时间戳目录，也可追加 `training.log_dir=/你的新目录`。`algo.max_iterations` 控制训练长度；上述命令保留任务默认的每 rank 2048 环境、8192 batch、每轮 8 次 critic / temperature 与 2 次 actor 更新。`uv@0.12.5` 用于匹配这里的调用方式，不代表历史实验已验证该 uv 版本。

### 使用报告中的仿真 CPU 池配置

默认 CPU 配置能启动训练，但不等同于报告最终性能。本报告服务器的双卡 GPU 0–1 使用两个各 32 物理核的仿真池，完整命令为：

```bash
pool0="$(seq -s, 8 39)"
pool1="$(seq -s, 48 79)"
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=0 \
uvx uv@0.12.5 run --no-sync \
  src/unilab/scripts/train_sac.py \
  task=g1_walk_flat/mujoco 'training.devices=[0,1]' \
  "training.dp_collector_cpu_ids=[[$pool0],[$pool1]]" \
  algo.max_iterations=1000 training.no_play=true training.trace_enabled=false
```

单卡本地池对照是在上述单卡训练命令中追加 `"+env.cpu_ids=[$pool0]"`，并使用物理 GPU 0。CPU 列表按 rank 顺序分配；换服务器或 GPU 组合时先检查 `lscpu -e=CPU,CORE,SOCKET,NODE` 和 `nvidia-smi topo -m`，重新选择并实测。不要在外层包 `taskset` / `numactl`，以免同时限制 learner/NCCL。

## 用本仓库复现实验

以下为历史实验的独立复测入口，需要技术报告记录的三个源码版本及优化代码、对应运行环境。它与上面的“仅使用 UniLab”训练入口分开；版本及差异见 [技术报告](docs/REPORT_zh.md)，这里不包含代码安装步骤。

```bash
export OPT_REPO="/你的路径/UniLab-Multi-Card-Training-Optimization"
export UNILAB_WORKSPACE="/你的路径/UniLabSim"
# 历史复测脚本要求该目录下有 UniLab/、unilab_rl/、unisim/。
export PYTHONPATH="$UNILAB_WORKSPACE/UniLab/src:$UNILAB_WORKSPACE/unilab_rl/src:$UNILAB_WORKSPACE/unisim/src"
export UNILAB_EXPERIMENT_DIR="$UNILAB_WORKSPACE/experiment_runs/reproduce"
cd "$UNILAB_WORKSPACE/UniLab"
unset CUDA_VISIBLE_DEVICES NCCL_PROTO NCCL_ALGO NCCL_GRAPH_MIXING_SUPPORT NCCL_GRAPH_STREAM_ORDERING

# 单卡本地32核 / 双卡各32核，交替三组；串行运行，使用新prefix。
uvx uv@0.12.5 run --no-sync python "$OPT_REPO/scripts/run_matrix.py" \
  n1_local32 n2_pool32 --repeats 3 --iterations 1000 --prefix repro

uvx uv@0.12.5 run --no-sync python "$OPT_REPO/scripts/analyze_run.py" \
  "$UNILAB_EXPERIMENT_DIR/repro_r1_n2_pool32" \
  --output "$UNILAB_EXPERIMENT_DIR/recomputed_metrics.json"

CUDA_VISIBLE_DEVICES=0,1 NCCL_SHM_DISABLE=0 uvx uv@0.12.5 run --no-sync python \
  "$OPT_REPO/scripts/dp_probe.py" correctness --compile-loss \
  --output "$UNILAB_EXPERIMENT_DIR/correctness.json"

uvx uv@0.12.5 run --no-project python "$OPT_REPO/scripts/verify_repository.py"
```

主指标是最后 50% TensorBoard 事件的累计总环境步数差 / 墙钟时间差，排除冷启动。历史顺序是先 `n1_total64 n2_pool32` 交替三组，再单独三次 `n1_local32`；上面的新复测采用交替本地池对照。脚本的 CPU 预设见 [CPU_POOLS](scripts/experiment_paths.py)。

## 仓库内容

- [技术报告](docs/REPORT_zh.md)：改动、设计依据、版本、参数、详细数据、验证与诊断复现。
- [补丁](patches/unilab_rl_optimization.patch)：3 个生产文件、4 个测试文件，面向技术报告记录的固定 RL 版本。
- [实验代码](scripts/)；[汇总](results/2026-10-03/summary.json)、[原始指标](results/2026-10-03/raw/)、[验证日志](results/2026-10-03/validation/)。
- [原始证据包](artifacts/sac_20261002_evidence.zip)与[哈希清单](artifacts/evidence_manifest.json)：335 项资料，保留历史脚本、日志、报告与 trace 供审计；日常使用以上两个文档。

实测于 2026-10-02～03；此次只整理文档，未重新跑性能测试。补丁涉及的上游源码继续适用其 [Apache-2.0 许可证](licenses/unilab_rl-LICENSE)。
