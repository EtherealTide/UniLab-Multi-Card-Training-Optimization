# UniLab 多卡 SAC 训练优化

在 8×RTX 5090 服务器上，对 **UniLab + MuJoCo + G1WalkFlat + SAC** 的真实训练进行双卡优化。仓库提供详细报告、可应用到 `unilab_rl` 的补丁、训练/诊断脚本和原始实验资料。

**主要结果：单卡 50,194 → 双卡 108,646 env steps/s，均值比 2.165×。** 每 rank 保持 `num_envs=2048`、`batch_size=8192`、每轮 8 次 critic / temperature 更新、2 次 actor 更新，保留 Inductor 和整轮 CUDA Graph。关键配置是 GPU 0–1、NCCL SHM，以及**只限制 MuJoCo 仿真线程池，不限制整个 learner 进程的 CPU 亲和性**。

> 这是包含 CPU 配额、collector 与线程池布局变化的端到端**弱扩展**结果，不是纯 GPU 算力扩展效率。双卡全局环境数/batch 为 4096/16384。每次测试 1000 iterations，统计后 50% 的累计环境步数 / 共同墙钟时间；尚未验证长期收敛。实测日期：2026-10-02～03；仓库整理日期：2026-10-06。

## 主要实验结果

| 配置 | 单卡平均 steps/s | 双卡平均 steps/s | 比值 | 重复次数 |
|---|---:|---:|---:|---|
| 图内 DP + SHM，自动 CPU 配置 | 30,766 | 52,750 | 1.715× | 各 3 次，交替 |
| 仿真池总共 64 个物理核：单池 64 / 双池各 32 | 45,510 | 108,646 | 2.387× | 各 3 次，交替 |
| **本地仿真池：单卡 32 / 双卡各 32** | **50,194** | **108,646** | **2.165×** | 各 3 次；单卡补充组随后运行 |

最终双卡三次为 **107,503、109,620、108,816 steps/s**；本地 32 核单卡三次为 **48,057、51,036、51,489 steps/s**。不使用日志瞬时 `Perf/total_fps` 的均值计算主加速比，因为该指标可能高估共同墙钟吞吐。

原始基线代码直接拒绝 NVIDIA 整轮 CUDA Graph 与 DP 并用，因此没有有效的“原始默认双卡 FPS”。关闭 graph mixing、强制 Simple 协议、把整个进程树绑到 NUMA 0 / 64 核均未选为推荐；完整负面数据见报告。

- [详细实验报告](docs/REPORT_zh.md)：分析过程、优化点、全部主要对照、Nsight / CUDA Event 证据和限制。
- [机器可读汇总](results/2026-10-03/summary.json) / [原始指标和命令](results/2026-10-03/raw/) / [验证日志](results/2026-10-03/validation/)。
- [完整补丁](patches/unilab_rl_optimization.patch)：3 个生产文件、4 个测试文件，包含新增文件。
- [原始证据包](artifacts/sac_20261002_evidence.zip)：约 9.6 MB，335 个清单项，含原始脚本、TensorBoard、日志、选定 Perfetto trace、Nsight report。训练 checkpoint 未打包。

## 优化内容

1. 允许 NCCL 梯度同步进入整轮 CUDA Graph，每个 optimizer step 前仍平均梯度，每轮保持 18 次 collective。
2. 在销毁 process group 之前释放捕获的图，修复训练完成后退出挂起。
3. 用随梯度一起归约的 finite-loss 标记，使任一 rank 遇到非有限 loss 时，各 rank 一致跳过更新，不增加 collective。
4. 修复 graph replay 的计数/计时；正常训练仅计数，诊断模式才启用 external CUDA Event；合并细碎的统计 payload 提交。
5. 利用现有 CPU 池接口调优：仿真 worker 使用物理核，learner/NCCL 保持原有亲和性。**CPU 编号属于本机配置，没有硬编码进算法库默认策略。**

在未调 CPU 池时，首次 critic 同步 event 区间约 22–25 ms，后续同尺寸同步约 0.55 ms。只调整仿真池后，同步 event 总和约从 30.21 降至 13.03 ms/rank，最终双卡 learner 均值约 23.55 ms。长 NCCL 区间包含等待，不能全部归因于 PCIe 带宽。

## 环境与固定基线

Linux；8×RTX 5090；双 Xeon Platinum 8380，80 物理核 / 160 逻辑线程；约 1 TB RAM。GPU 0–1 / 4–5 为 PIX，0–4 为跨 NUMA SYS。实测 GPU 0–1 peer access 不可用；默认禁用 SHM 时走 NET/IB，启用后确认 SHM/direct。

| 组件 | 实验版本 / commit |
|---|---|
| [UniLab](https://github.com/unilabsim/UniLab) | `0fd2bd5d72210ad685838ade4ba4687b269f7de2` |
| [unilab_rl](https://github.com/unilabsim/unilab_rl) | `e2f18b4df1dcb0c35cd4c2dccd0517151633e53d`（1.4.4） |
| [unisim](https://github.com/unilabsim/unisim) | `783220d64ba7a207d316f1dd06ccd1a1db1261bc` |
| PyTorch / CUDA / NCCL | `2.14.0+cu130` / 13.0 / 2.30.7 |
| Driver / MuJoCo | 580.95.05 / 3.11.0 |

## 在真实 UniLab 项目中启用

以下命令在 **Linux GPU 服务器**执行。本仓库不替代三个上游项目；已有可运行的 UniLab 环境可直接从第 2 步开始。新环境安装与版本核对见 [复现说明](docs/REPRODUCIBILITY.md)。

### 1. 设置路径

```bash
git clone https://github.com/EtherealTide/UniLab-Multi-Card-Training-Optimization.git
export OPT_REPO="$(realpath UniLab-Multi-Card-Training-Optimization)"
export UNILAB_WORKSPACE="$HOME/desktop/UniLabSim"
# UNILAB_WORKSPACE 中应有 UniLab/、unilab_rl/、unisim/ 三个源码目录。
```

### 2. 应用补丁并确保导入修改后的源码

补丁针对上表 `unilab_rl` commit。已有补丁的实验服务器无需重复应用。新版本先做 `--check`，不要忽略冲突或丢弃自己的修改。

```bash
git -C "$UNILAB_WORKSPACE/unilab_rl" rev-parse HEAD
git -C "$UNILAB_WORKSPACE/unilab_rl" apply --check "$OPT_REPO/patches/unilab_rl_optimization.patch"
git -C "$UNILAB_WORKSPACE/unilab_rl" apply "$OPT_REPO/patches/unilab_rl_optimization.patch"

cd "$UNILAB_WORKSPACE/UniLab"
export PYTHONPATH="$UNILAB_WORKSPACE/UniLab/src:$UNILAB_WORKSPACE/unilab_rl/src:$UNILAB_WORKSPACE/unisim/src"
uv run --no-sync python -c 'import uni_rl; print(uni_rl.__file__)'
# 输出应指向工作区 unilab_rl/src，而不是未修改的 site-packages 副本。
```

### 3. 启动优化后的双卡真实训练

```bash
uv run --no-sync python "$OPT_REPO/scripts/run_optimized.py" \
  --cards 2 --iterations 1000 \
  --log-dir "$UNILAB_WORKSPACE/experiment_runs/optimized_dual_01"

# 单卡对照：相同本地32核仿真池、相同每rank训练参数。
uv run --no-sync python "$OPT_REPO/scripts/run_optimized.py" \
  --cards 1 --single-pool local32 --iterations 1000 \
  --log-dir "$UNILAB_WORKSPACE/experiment_runs/optimized_single_01"
```

输出目录必须未存在。可先加 `--dry-run` 查看命令。启动器固定使用物理 GPU 0–1、设置 `NCCL_P2P_DISABLE=1` / `NCCL_SHM_DISABLE=0`，清除实验用的 NCCL 算法/协议/graph 调度覆盖；默认关闭 trace。本机仿真 CPU 池为 rank 0：`8–39`，rank 1：`48–79`。

**不要在启动器外层加 `taskset` / `numactl`。** 它们会限制 learner/NCCL，在本次实验中反而更慢。迁移到其他服务器时先检查 `lscpu -e=CPU,CORE,SOCKET,NODE` 和 `nvidia-smi topo -m`，调整 [CPU_POOLS](scripts/experiment_paths.py)；这些编号不是通用硬件自动发现策略。

### 不使用启动器，直接传给 UniLab

补丁和 `PYTHONPATH` 同上。在 UniLab 根目录运行下面的 Bash 命令（`seq` 生成完整 Hydra 列表，不是省略号）：

```bash
pool0="$(seq -s, 8 39)"
pool1="$(seq -s, 48 79)"
CUDA_VISIBLE_DEVICES=0,1 NCCL_SHM_DISABLE=0 uv run --no-sync python \
  src/unilab/scripts/train_sac.py \
  task=g1_walk_flat/mujoco 'training.devices=[0,1]' \
  "training.dp_collector_cpu_ids=[[$pool0],[$pool1]]" \
  training.no_play=true training.trace_enabled=false algo.max_iterations=1000 \
  "training.log_dir=$UNILAB_WORKSPACE/experiment_runs/direct_dual_01"
```

这使用真实 UniLab task、环境和训练入口，没有替换成合成环境。`training.dp_collector_cpu_ids` 作用于 DP collector；单卡对应接口是 `+env.cpu_ids=[...]`。默认训练命令不会自动获得本机 CPU 池配置，必须显式应用。

## 复现实验和验证

```bash
cd "$UNILAB_WORKSPACE/UniLab"
export UNILAB_EXPERIMENT_DIR="$UNILAB_WORKSPACE/experiment_runs/reproduce"
unset CUDA_VISIBLE_DEVICES NCCL_PROTO NCCL_ALGO NCCL_GRAPH_MIXING_SUPPORT NCCL_GRAPH_STREAM_ORDERING

# 方便新增复测：单卡本地32核 / 双卡各32核，交替三组。
uv run --no-sync python "$OPT_REPO/scripts/run_matrix.py" \
  n1_local32 n2_pool32 --repeats 3 --iterations 1000 --prefix repro_local

# 精确对应原报告最终矩阵的顺序：先64核单卡与双卡交替，再补单卡本地32核。
uv run --no-sync python "$OPT_REPO/scripts/run_matrix.py" \
  n1_total64 n2_pool32 --repeats 3 --iterations 1000 --prefix repro_pool
uv run --no-sync python "$OPT_REPO/scripts/run_matrix.py" \
  n1_local32 --repeats 3 --iterations 1000 --prefix repro_local_control

# 真实 NCCL + CUDA Graph 的小模型正确性（不是性能基线）。
CUDA_VISIBLE_DEVICES=0,1 NCCL_SHM_DISABLE=0 uv run --no-sync python \
  "$OPT_REPO/scripts/dp_probe.py" correctness --compile-loss \
  --output "$UNILAB_EXPERIMENT_DIR/correctness.json"

# 重算一个训练目录的尾段指标。
uv run --no-sync python "$OPT_REPO/scripts/analyze_run.py" \
  "$UNILAB_EXPERIMENT_DIR/repro_local_r1_n2_pool32" \
  --output "$UNILAB_EXPERIMENT_DIR/recomputed_metrics.json"
```

每次使用新的 prefix；性能任务串行执行。启动/编译耗时单独保存，主指标不包含冷启动。更多通信、CUDA Event、Nsight 命令见 [复现说明](docs/REPRODUCIBILITY.md)。

原补丁验证：**388 CPU tests passed / 45 skipped / 3 deselected；32 CUDA tests passed；mypy/Pyright/Ruff 通过**。真实双 rank probe 验证参数与 optimizer 状态一致、warmup 不改变状态/RNG、重捕获计数正确、单 rank Inf loss 时三种 optimizer 一致跳步。

仓库资料自检不需要 GPU 或第三方 Python 包：

```bash
uv run --no-project python "$OPT_REPO/scripts/verify_repository.py"
```

## 文件布局与来源

```text
docs/            详细报告、复现说明、来源说明
patches/         最终完整补丁（保持原始字节）
scripts/         支持显式工作区路径的当前复现入口
results/         汇总、原始 JSON、验证日志
artifacts/       不改写的原始证据包及逐文件哈希清单
```

原始报告中的服务器路径是历史记录，当前可复制命令以本 README 为准。归档内保留实验当时的脚本和路径；`scripts/` 是为独立仓库整理后的入口，增加路径配置/dry-run/案例选择，不改变训练算法或重新生成实测数字。参见 [来源与许可说明](docs/PROVENANCE.md)。
