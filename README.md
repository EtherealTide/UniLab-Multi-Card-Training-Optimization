# UniLab 双卡 SAC 训练优化

在真实 **UniLab / MuJoCo / G1WalkFlat / SAC** 训练中，RTX 5090 单卡平均 **50,194**、双卡 **108,646 env steps/s**，均值比 **2.165×**。保留 Inductor + 整轮 CUDA Graph，每 rank 使用 2048 环境、8192 batch、每轮 8 次 critic / temperature 更新与 2 次 actor 更新。

| 配置（各 3 次） | 单卡 steps/s | 双卡 steps/s | 均值比 |
|---|---:|---:|---:|
| 图内梯度同步 + SHM，自动 CPU 配置 | 30,766 | 52,750 | 1.715× |
| 仿真池共 64 核：单池 64 / 双池各 32 | 45,510 | 108,646 | 2.387× |
| **本地仿真池：单卡 32 / 双卡各 32** | **50,194** | **108,646** | **2.165×** |

结论：图内 all-reduce 可以与整轮 CUDA Graph 共存；实际瓶颈还包括 CPU 仿真、IPC 和跨 rank 等待。推荐 GPU 0–1、NCCL SHM，以及仅设置仿真线程池 CPU 列表，保持 learner 原有亲和性。原始默认双卡因代码保护直接报错，没有有效基线 FPS。

这是 CPU 配额、collector 和 GPU 一起扩展的**端到端弱扩展**，不是补丁独立收益或纯 GPU 扩展效率。双卡全局环境数 / batch 加倍；尚未验证长期收敛。全部版本、参数、逐次数据、分析与限制见 [技术报告](docs/REPORT_zh.md)。

## 复现与实际使用

以下在 Linux GPU 服务器执行，需要 Git、uv 和可运行 MuJoCo 的 UniLab 环境。已有三个源码仓库时跳过克隆步骤；复现历史结果需使用固定版本，勿重置已有修改。

### 1. 准备源码与环境

```bash
git clone https://github.com/EtherealTide/UniLab-Multi-Card-Training-Optimization.git
export OPT_REPO="$(realpath UniLab-Multi-Card-Training-Optimization)"
export UNILAB_WORKSPACE="$HOME/desktop/UniLabSim"

# 仅用于新的空工作区。
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

依赖和资产按固定版本上游要求安装；`uv sync` 不保证重建历史二进制环境。对照报告核对版本与拓扑：

```bash
nvidia-smi topo -m
lscpu -e=CPU,CORE,SOCKET,NODE
uv run --no-sync python -c 'import torch,mujoco; print(torch.__version__,torch.version.cuda,torch.cuda.nccl.version(),mujoco.__version__)'
```

### 2. 应用补丁，启用修改后的算法库

```bash
git -C "$UNILAB_WORKSPACE/unilab_rl" apply --check "$OPT_REPO/patches/unilab_rl_optimization.patch"
git -C "$UNILAB_WORKSPACE/unilab_rl" apply "$OPT_REPO/patches/unilab_rl_optimization.patch"
export PYTHONPATH="$UNILAB_WORKSPACE/UniLab/src:$UNILAB_WORKSPACE/unilab_rl/src:$UNILAB_WORKSPACE/unisim/src"
uv run --no-sync python -c 'import uni_rl; print(uni_rl.__file__)'
```

导入路径应指向工作区 `unilab_rl/src`。已应用补丁的实验服务器无需重复应用；新版上游若检查失败，应先解决兼容性。

### 3. 在真实 UniLab 中训练

```bash
uv run --no-sync python "$OPT_REPO/scripts/run_optimized.py" \
  --cards 2 --iterations 1000 \
  --log-dir "$UNILAB_WORKSPACE/experiment_runs/dual_01"
```

启动器调用真实 `UniLab/src/unilab/scripts/train_sac.py`，使用 `task=g1_walk_flat/mujoco`、`training.devices=[0,1]`，设置 `NCCL_P2P_DISABLE=1`、`NCCL_SHM_DISABLE=0`；不改变任务的 batch、环境数或更新频率。单卡对照使用 `--cards 1 --single-pool local32` 和新的输出目录；加 `--dry-run` 可查看完整 Hydra 命令，直接用于自己的训练入口。

本机双卡仿真池通过 `training.dp_collector_cpu_ids` 分别设置 CPU `8–39`、`48–79`；单卡使用 `+env.cpu_ids`。迁移服务器时修改 [CPU_POOLS](scripts/experiment_paths.py)，按物理核和 NUMA 拓扑重新选择。不要照搬编号或在启动器外包 `taskset` / `numactl`；本次实验中限制整个 learner 进程树会降低吞吐。

### 4. 重复实验与检查

```bash
export UNILAB_EXPERIMENT_DIR="$UNILAB_WORKSPACE/experiment_runs/reproduce"
unset CUDA_VISIBLE_DEVICES NCCL_PROTO NCCL_ALGO NCCL_GRAPH_MIXING_SUPPORT NCCL_GRAPH_STREAM_ORDERING

# 单卡本地32核 / 双卡各32核，交替三组；串行运行，使用新prefix。
uv run --no-sync python "$OPT_REPO/scripts/run_matrix.py" \
  n1_local32 n2_pool32 --repeats 3 --iterations 1000 --prefix repro

uv run --no-sync python "$OPT_REPO/scripts/analyze_run.py" \
  "$UNILAB_EXPERIMENT_DIR/repro_r1_n2_pool32" \
  --output "$UNILAB_EXPERIMENT_DIR/recomputed_metrics.json"

CUDA_VISIBLE_DEVICES=0,1 NCCL_SHM_DISABLE=0 uv run --no-sync python \
  "$OPT_REPO/scripts/dp_probe.py" correctness --compile-loss \
  --output "$UNILAB_EXPERIMENT_DIR/correctness.json"

uv run --no-project python "$OPT_REPO/scripts/verify_repository.py"
```

主指标是最后 50% TensorBoard 事件的累计总环境步数差 / 墙钟时间差，排除冷启动，不用瞬时 FPS 均值。矩阵保存命令、版本、配置和结果。历史运行顺序是先 `n1_total64 n2_pool32` 交替三组，再单独运行三次 `n1_local32`；上面的新复测采用交替本地池对照。

## 仓库内容

- [技术报告](docs/REPORT_zh.md)：改动、设计依据、版本、参数、详细数据、验证与诊断复现。
- [补丁](patches/unilab_rl_optimization.patch)：3 个生产文件、4 个测试文件，面向上述固定 RL 版本。
- [实验代码](scripts/)；[汇总](results/2026-10-03/summary.json)、[原始指标](results/2026-10-03/raw/)、[验证日志](results/2026-10-03/validation/)。
- [原始证据包](artifacts/sac_20261002_evidence.zip)与[哈希清单](artifacts/evidence_manifest.json)：335 项资料，保留历史脚本、日志、报告与 trace 供审计；日常使用以上两个文档。

实测于 2026-10-02～03；此次只整理文档，未重新跑性能测试。补丁涉及的上游源码继续适用其 [Apache-2.0 许可证](licenses/unilab_rl-LICENSE)。
