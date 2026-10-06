# UniLab SAC 双 RTX 5090 训练优化实验报告

> GitHub 整理版：数值与原始报告一致。本文第8节中的服务器路径为历史实验路径；在新工作区复现请使用[仓库 README](../README.md)和[当前复现说明](REPRODUCIBILITY.md)。原始报告、脚本及完整证据保存在[归档](../artifacts/sac_20261002_evidence.zip)；当前便携脚本位于 [scripts/](../scripts/)。

实验日期：2026-10-02 至 2026-10-03（Asia/Singapore）

任务：G1WalkFlat / MuJoCo / SAC；服务器：`5090-server2`
本文包含源码优化、CUDA Event / Nsight 分析、成功与失败对照，以及可复现的本机运行配置。

## 1. 已确认的结果与适用范围

已经使原先明确拒绝多卡的 FastSAC NVIDIA 整轮 CUDA Graph 路径支持两卡梯度同步，并保留 Inductor 编译、整轮 graph replay，以及每次优化器更新前的梯度平均。没有通过减少 critic/actor 更新次数获得吞吐量提升。

最终推荐配置在 GPU 0–1 上三次达到 **107,503 / 109,620 / 108,816 env steps/s**，平均 **108,646 steps/s**。关键增益来自保留整轮编译图、启用 SHM，以及**只给 MuJoCo 仿真线程池分配物理核，保持 learner 进程原有 CPU 亲和性**。环境数、batch 和更新频率保持指定默认值。

更稳妥的单卡基线使用与一个 DP rank 相同的本地 32 核仿真池，三次平均 **50,194 steps/s**；最终端到端比值为 **2.165×**。另一个固定总计 64 个仿真物理核的单卡对照平均为 **45,510 steps/s**，比值 **2.387×**。两种比较都包含线程池、NUMA 和并行 collector 布局的变化，不能解释为纯 GPU 算力扩展效率。所有主指标均以真实累计环境步数除以共同墙钟窗口计算。

在优化 CPU 池之前，自动线程配置下三组单卡 / 双卡均值分别为 **30,766 / 52,750 steps/s**，即 **1.715×**。因此，最终双卡比同一代码的自动 CPU 配置还快 **2.060×**；这些阶段应分开解释，不能把全部提升归因于 all-reduce 优化。

先前一组探索实验曾达到 57,031.46 steps/s，对应同阶段单卡的 1.833×。本报告同时保留这些中间结果、负面调优与最终重复实验。

主要发现如下：

1. **All-reduce 不必迫使整轮 CUDA Graph 退回 eager。** 在本机软件版本和一进程一卡布局下，NCCL collective 可以进入整轮图；真正需要补齐的是图生命周期、跨 rank 数值异常处理和 replay 计时。
2. **本机实际没有可用的 GPU 0–1 CUDA peer access。** PIX 表示 PCIe 路径接近，不等于可用 P2P。默认配置禁用 SHM 后，NCCL 日志显示实际走 NET/IB；显式启用 SHM 后确认走 SHM/direct。
3. **双卡 learner 的主要额外耗时集中于每轮第一个 critic collective。** 独立诊断中它约 22.8–25.5 ms，后续同量级 critic collective 约 0.52–0.56 ms。不能把第一个 collective 的全部时间解释为数据传输；其中包含同步、调度或等待。
4. **端到端训练还受 CPU 环境推进、重置、IPC 与流水线等待影响。** 单卡 learner 仅约 11 ms，而环境推进约 60 ms；仅优化矩阵计算不能保证端到端线性扩展。

本文验证的是吞吐量和运行时正确性，不是相同样本预算下的收敛速度或最终策略质量。每 rank 保持 2048 环境、8192 batch，因此双卡是全局 4096 环境、16384 batch 的弱扩展实验；它与单卡的优化轨迹不完全相同。

## 2. 环境、版本与实验边界

### 2.1 硬件与通信拓扑

| 项目 | 实测或服务器信息 |
|---|---|
| GPU | 8 × NVIDIA GeForce RTX 5090，单卡 32607 MiB |
| CPU | 双 Intel Xeon Platinum 8380，80 物理核 / 160 逻辑线程 |
| 内存 | 约 1 TB；`/dev/shm` 约 504 GB |
| GPU 0–1 / 4–5 | PIX，同 NUMA，优先测试组合 |
| GPU 0–4 | SYS，跨 NUMA 对照组合 |
| NUMA 0 CPU | 0–39、80–119 |
| NUMA 1 CPU | 40–79、120–159 |
| GPU 0–1 peer access | `torch.cuda.can_device_access_peer(0,1) == False` |
| 默认 NCCL 路径 | P2P 禁用、SHM 禁用；INFO 日志确认 NET/IB |
| SHM 实验路径 | `NCCL_SHM_DISABLE=0`；INFO 日志确认 SHM/direct |

GPU 0–1 和 4–5 是合理的优先组合，但最终选择仍以真实训练和通信实测为准。不能仅由 `nvidia-smi topo -m` 推断带宽、延迟或 P2P 可用性。

### 2.2 软件与仓库版本

按用户要求，服务器三个仓库的原有本地修改已放弃并更新。实验根目录为 `/mnt/gfs/home/liangyu/desktop/UniLabSim`。

| 项目 | 实验基线 |
|---|---|
| UniLab | `0fd2bd5d72210ad685838ade4ba4687b269f7de2` |
| unilab_rl | `e2f18b4df1dcb0c35cd4c2dccd0517151633e53d`，版本 1.4.4 |
| unisim | `783220d64ba7a207d316f1dd06ccd1a1db1261bc` |
| PyTorch | `2.14.0+cu130` |
| PyTorch CUDA | 13.0 |
| NCCL | 2.30.7，安装包标识 CUDA 13.3 |
| NVIDIA driver | 580.95.05 |
| MuJoCo | 3.11.0 |

性能代码改动集中在 unilab_rl。最终状态检查确认 UniLab 和 UniSim 工作区均干净，unilab_rl 仅有 3 个生产文件及 4 个测试文件改动；没有破坏 `uni_rl` 不导入 `unilab` / `unisim`、环境由调用方注入的边界。

运行时显式设置三个仓库 `src` 的 `PYTHONPATH`，避免虚拟环境内已安装的副本遮蔽源码改动。Python 命令使用 `uv run --no-sync`，实验期间没有隐式升级依赖。

### 2.3 保持不变的训练参数

| 参数 | 每 rank 配置 |
|---|---:|
| task / backend | `g1_walk_flat/mujoco` |
| num_envs | 2048 |
| batch_size | 8192 |
| updates_per_step | 8 |
| policy_frequency | 4，即每轮 2 次 actor update |
| critic / temperature updates | 每轮各 8 次 |
| gradient collectives | 每轮 8 + 8 + 2 = 18 次 |
| seed | 1 |
| 正式对照长度 | 1000 iterations |
| logger / log_interval | TensorBoard / 1 |
| replay prefetch | `one_tick` |
| use_compile / AMP | 保持任务默认配置启用 |

训练使用真实 G1 环境推进、奖励、重置、replay 与参数更新。小模型合成数据只用于正确性和通信隔离实验，不作为训练吞吐量的替代。

## 3. 吞吐量定义与实验方法

### 3.1 主指标：共同墙钟窗口内的总环境步数

每次训练提取 TensorBoard 后 50% 的 scalar 记录，主指标为：

```text
aggregate_env_steps_per_second
  = (last_event.step - first_event.step)
    / (last_event.wall_time - first_event.wall_time)
```

事件 step 使用训练记录的累计总环境步数。双卡主日志记录跨 rank 聚合后的总步数；两个设备不分别计时后再相加。这个窗口覆盖迭代之间的统计同步与日志开销，并排除首次编译、环境初始化以及大部分热身。

这是**稳态尾段吞吐量**，并不包括整个进程从启动到退出的所有时间；首次 Inductor 编译可能消耗数分钟，短任务不能用此指标估计总完成时间。每次实验同时记录进程墙钟时间和退出码，以免把结束阶段挂死的运行当作成功完成。

### 3.2 为什么不能直接使用 `Perf/total_fps`

现有 `Perf/total_fps` 汇总各 rank 的速率，其迭代计时边界还没有覆盖随后发生的全部统计与日志工作。平均瞬时速率也不等于总工作量除以总时间。在双卡中，两种口径差异明显：最终第二组日志均值为 69,423.46 FPS，而共同墙钟实测为 54,105.31 steps/s。

因此报告保留日志 FPS 用于诊断，但所有加速比使用共同墙钟吞吐量。某一轮 `final_env_steps_per_sec` 或某个峰值不能作为整段训练结论。

### 3.3 对照与诊断

性能运行串行执行，避免本任务的多个 GPU 作业相互竞争。正式候选采用单卡 / 双卡交替运行三组，避免只挑选最快的一次。诊断运行单独开启 trace 与 CUDA event；它们不与无 trace 的正式重复实验混成一个均值。

使用的证据包括：

- 真实训练 `run_config.json`、`run_summary.json`、TensorBoard event 文件。
- `*_execution.json` 中的命令、环境覆盖、仓库 SHA、代码 diff 哈希、退出码及进程时间。
- Perfetto 阶段 trace 和每次 collective 的 external CUDA event。
- NCCL INFO 传输路径日志。
- 合成双 rank 模型与优化器状态一致性检查、通信 microbenchmark。

阶段存在流水线重叠，CPU span 也可能包含等待。因此不能把所有阶段均值直接相加，也不能把 CUDA event 跨 collective 的时间等同于 NIC/PCIe 纯传输时间。

## 4. 分析与实现迭代

### 4.1 原始路径为什么无法作为双卡基线

基线 FastSAC 的 NVIDIA 路径使用 Inductor 和整轮 CUDA Graph。`set_gradient_sync` 明确拒绝该路径绑定 DP callback，抛出：

```text
FastSAC NVIDIA CUDA whole-cycle mode does not support DP fallback
```

所以原始 main 的默认双卡 SAC 是功能失败，不存在可直接比较的有效原始双卡 FPS。报告中的双卡数字都来自启用图内梯度同步后的版本，不能描述为“在原始双卡吞吐量上提升了某个百分比”。

### 4.2 保留整轮图并捕获梯度同步

解除互斥限制，让 pack → all-reduce → average → unpack 与对应优化器更新一起进入整轮图。仍在每次 optimizer step 前进行跨 rank 梯度平均，保持每轮 18 次同步，不做延迟若干轮同步，也不以减少更新来换吞吐量。

NCCL 官方文档明确提供 CUDA Graph 支持，但所有 rank 必须一致地参与 collective 的捕获和执行。本次一进程一卡的实测验证了这一具体配置可用，并不意味着任意 NCCL / CUDA / PyTorch 组合都无需再测。[NCCL CUDA Graph 文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html)

### 4.3 处理图与 process group 的销毁顺序

早期实验已经完成训练，却在退出时挂起。修复为先解绑 learner 梯度同步 callback、释放捕获的图，再关闭 DP process group。IPC 清理异常也通过 `finally` 保证走到该顺序。

这一点是功能完整性要求，不只是性能优化。NVIDIA 也说明仍存活的图可能使通信资源销毁挂起。[CUDA Graph 进程挂起说明](https://docs.nvidia.com/dl-cuda-graph/troubleshooting/process-hang.html)

### 4.4 SHM 与传输路径

原配置同时禁用了 P2P 与 SHM。在本机 P2P 不可用的条件下，默认实际通过 NET/IB 通信。设置 `NCCL_SHM_DISABLE=0` 后日志确认 SHM/direct，并在探索阶段降低 learner 耗时。

这里只建议在已验证的这台服务器上显式设置该环境变量；没有把其他机器的保守默认值强行改成 SHM。早期名称含 `socket` 的运行实际不能据此称为 socket 传输，判断以 NCCL 日志为准。[NCCL 环境变量文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

### 4.5 修正 CUDA Graph replay 的同步计数与计时

旧统计代码只在 Python callback 执行时累加。在 graph replay 中 Python 不再逐次执行，导致 warmup/capture 的次数与耗时进入日志，实际 replay 则没有正确计量。早期 36 次调用及约 82,155 ms 的同步记录因此不能用于通信分析。

新增 capture / replay 生命周期钩子：捕获时记录图内固定 collective 数量；每次 replay 计入真实调用次数；诊断模式使用 `external=True` CUDA event 测量重放过程。图重建时清理旧事件，batch shape 或更新次数变化后重新捕获也能正确计数。

正常训练不启用 event 计时，只计调用次数，并省略不可用的毫秒指标，避免将“未测量”伪装成零耗时。trace 同时启用 CUDA events 时才输出图内同步时长。

### 4.6 保证跨 rank 的 finite gating 一致

只有梯度 all-reduce 并不足以处理所有数值异常。例如某 rank 的 loss 为 Inf，梯度却仍为有限值时，若依据各自 loss 决定是否跳过 optimizer，会导致各 rank 参数和 Adam 状态分叉。

针对 CUDA Graph / 无 GradScaler 路径，增加常驻 float32 finite sentinel，将本 rank 的非有限 loss 标记附带到原有梯度 all-reduce。平均后的任意正标记使所有 rank 一致跳过该优化器更新；仍结合梯度有限性检查。没有增加第 19 次 collective。sentinel 不属于优化器参数，也不进入 checkpoint。

这项改动服务于正确性，不应单独归因为性能收益。

### 4.7 减少统计 payload 的细碎提交

DP 统计 payload 从逐个标量填充设备 tensor 改为构建一次 host payload，再以单次 tensor 创建 / 传输提交。减少每轮细碎 CPU→GPU 操作，但没有对该改动做独立重复消融，因此不能声称它单独带来了某个百分比。

## 5. 性能实测

### 5.1 中间阶段：自动 CPU 配置的三组交替重复

各行均为 1000 iterations、正常 trace 关闭、训练成功退出；双卡使用 GPU 0–1 和 SHM。

| 组别 | 单卡实际 steps/s | 双卡实际 steps/s | 双卡/单卡 | 单卡 learner ms | 双卡 learner ms |
|---|---:|---:|---:|---:|---:|
| 1 | 31,008.18 | 54,089.93 | 1.744× | 11.079 | 41.487 |
| 2 | 30,708.93 | 54,105.31 | 1.762× | 11.382 | 41.531 |
| 3 | 30,580.36 | 50,054.20 | 1.637× | 11.334 | 45.367 |
| 均值 | **30,765.82** | **52,749.81** | **1.715×（均值比）** | 11.265 | 42.795 |
| 样本标准差 | 219.51 | 2,334.49 | — | — | — |

第三次双卡明显较慢，应保留而不是删掉。三次样本仍很少，这些数据不足以提供严格的长期性能置信区间，也不能把差异全部归因于通信；CPU 调度、环境轨迹及重置负载等均可能影响端到端结果。

对应日志 `Perf/total_fps` 均值分别为：单卡 33,161.36 / 32,613.61 / 32,434.71；双卡 68,840.86 / 69,423.46 / 62,342.24。直接用这些数相除会夸大用户关心的实际加速比。

### 5.2 探索阶段与拓扑对照

下面各行软件阶段或诊断设置不完全相同，主要用于解释决策，不能视为严格单因素消融。

| 运行名 | 迭代 | 实际 steps/s | learner ms | 说明 |
|---|---:|---:|---:|---|
| `baseline_n1` | 200 | 约 29,940 | 11.210 | 原始单卡，短基线 |
| `graph_socket_abs` | 200 | 44,059.11 | 56.999 | 实际 NET/IB；训练结束后退出挂起 |
| `transport_n2_shm` | 200 | 49,681.69 | 42.500 | SHM；同样存在退出问题 |
| `cleanup_n2_shm` | 200 | 46,465.29 | 47.254 | 生命周期修复，退出码 0 |
| `longer_n1` | 1000 | 31,117.68 | 11.299 | 无 trace 单卡 |
| `longer_n2_shm` | 1000 | 57,031.46 | 38.197 | 早期较快双卡，1.833×，单次探索 |
| `optimized_n1` | 1000 | 30,285.28 | 11.104 | 中间版本单卡 |
| `optimized_n2_shm` | 1000 | 53,914.14 | 41.645 | GPU 0–1，中间版本含 event 计量 |
| `optimized_n2_45` | 1000 | 53,876.87 | 41.513 | GPU 4–5 |
| `optimized_n2_04` | 1000 | 49,592.96 | 45.093 | GPU 0–4，跨 NUMA |

在这组中间版本单次对照中，两个 PIX 组合吞吐量非常接近，跨 NUMA 组合比 GPU 0–1 低约 8%。它支持优先选 0–1 或 4–5，但还不足以区分两个 PIX 组合的长期优劣。

### 5.3 图调度与 NUMA 对照

以下运行均为真实训练 1000 iterations、trace 开启、GPU 0–1、SHM，且退出码为 0。它们用于诊断，不与第 5.1 节无 trace 的正式重复实验合并统计。

| 运行 | 相对默认的变化 | 实际 steps/s | learner ms |
|---|---|---:|---:|
| `nomix_clean_n2_shm` | `NCCL_GRAPH_MIXING_SUPPORT=0` | 54,472.57 | 40.654 |
| `graph_order_n2_shm` | 上述设置，再加 `NCCL_GRAPH_STREAM_ORDERING=0` | 45,089.22 | 48.558 |
| `numa0_n2_shm` | 默认图调度，`numactl --cpunodebind=0 --membind=0` | 42,229.65 | 52.651 |

早期 `nomix` 探索约 55,073.94 steps/s、learner 39.91 ms，但当时存在 CPU 测试套件并发，因此只记录为探索结果；`nomix_clean` 在没有该干扰的条件下重跑。

这些单次对照没有提供足够证据推荐替换默认设置。关闭 graph mixing 的提升未建立重复统计，关闭 graph stream ordering 和将整个进程树限制于 NUMA 0 在本次测试中反而明显变慢。NUMA 0 绑定同时改变可用 CPU 数量及调度范围，不能把其结果简单归因为“本地内存更慢”。

关闭 graph mixing 支持也存在适用条件，本次没有将这些开关写入全局默认配置。额外对照结果如下，均保持 2048 环境、8192 batch、8/2 次更新：

| 对照 | 单卡实际 steps/s | 双卡实际 steps/s | 结论 |
|---|---:|---:|---|
| 整个进程树 `taskset 8-39,48-79`，trace 开启 | 30,036.85 | 44,940.14 | 1.496×；learner/NCCL 也受限，效果较差 |
| 仅仿真池，每 rank 32 个物理核，trace 开启 | — | 109,565.41 | 明显改善，进入正式复测 |
| `NCCL_PROTO=Simple`，自动 CPU 配置，trace 开启 | — | 53,523.60 | 未见收益，不作为推荐 |

Simple 协议的真实双 rank 正确性 probe 通过；3.94 MB raw all-reduce 中位数约 487.63 µs，与自动协议的 487.95 µs 接近。没有把协议开关写入默认值。Nsight 检查见第 6.5 节。

### 5.4 最终配置：只控制仿真池，三次交替复测

单卡池使用 CPU `8–39,48–79`，共 64 个物理核；双卡的两个 collector 分别使用 `8–39` 与 `48–79`，每 rank 32 核、合计仍为 64 核。仿真 worker 不使用 SMT 兄弟线程，learner/NCCL 进程不施加 `taskset` 或 `numactl` 限制。两种配置都保留默认 Torch 线程预算。每次 1000 iterations、trace 关闭、退出码 0。

| 组别 | 单卡 64 核池 steps/s | 双卡 2×32 核池 steps/s | 比值 | 单卡 learner ms | 双卡 learner ms |
|---|---:|---:|---:|---:|---:|
| 1 | 42,506.53 | 107,503.16 | 2.529× | 13.268 | 21.650 |
| 2 | 46,352.36 | 109,619.53 | 2.365× | 11.878 | 24.423 |
| 3 | 47,670.96 | 108,815.94 | 2.283× | 12.206 | 24.581 |
| 均值 | **45,509.95** | **108,646.21** | **2.387×** | 12.451 | **23.552** |
| 样本标准差 | 2,683.29 | 1,068.35 | — | — | — |

这个对照控制了仿真池的总物理核数量，但单卡池跨两个 NUMA 节点，双卡则为两个各自局部的池，collector 数量也不同。因此它是部署配置的端到端比较，不是隔离 GPU 数量的单因素实验。

### 5.5 单卡本地 32 核对照

为了避免以跨 NUMA 的单卡池作为唯一基线，单卡另使用 `+env.cpu_ids=[8,...,39]`，与双卡 rank 0 的仿真 CPU 池完全相同，learner 仍不绑核。三个 1000-iteration 运行关闭 trace、均退出 0：

| 单卡复测 | 实际 steps/s | learner ms |
|---|---:|---:|
| 1 | 48,057.33 | 12.994 |
| 2 | 51,036.37 | 13.141 |
| 3 | 51,488.87 | 13.255 |
| 均值 | **50,194.19** | 13.130 |
| 样本标准差 | 1,864.36 | — |

最终双卡均值 / 此单卡均值 = **108,646.21 / 50,194.19 = 2.165×**。即便仅比较三次中最慢双卡与最快单卡，也为 2.088×；这只是这六次样本的范围，不是长期性能下界。

本地 32 核单卡对照在前述交替矩阵之后连续执行，不能称为与双卡交错配对的重复实验。这里每个 rank 的环境数、batch、仿真核数一致，双卡总仿真核数从 32 增至 64，是**GPU、collector 与 CPU 配额一起扩展的弱扩展比较**。超过 2× 的现象还可能包含任务轨迹、线程池效率和流水线重叠差异；不据此推导纯 GPU 超线性加速或收敛加速。

## 6. 瓶颈证据：计算、通信与等待

### 6.1 单卡不是纯 learner 算力瓶颈

短基线单卡 learner 约 11.21 ms，collector 环境 step 约 60.80 ms，其中后端 32.14 ms、状态更新 12.82 ms、完成环境重置 12.96 ms。后续单卡 1000 轮 learner 稳定在约 11.1–11.4 ms。

因此，将 learner 计算再减半，也不会自动把整体训练加速一倍。异步 collector / learner 的流水线效率和 CPU 环境推进同样重要。

### 6.2 双卡整体阶段

`optimized_n2_shm` 的尾段均值：

| 指标 | ms | 解释 |
|---|---:|---|
| learner 更新 | 41.65 | 包含图内同步相关等待 |
| 图内梯度同步 event 总和 / rank | 30.21 | 18 次 collective 的计时区间，非纯传输 |
| collector 环境 step | 40.14 | 与 learner 工作可能重叠 |
| MuJoCo 后端 | 24.82 | 环境 step 的子阶段 |
| learner 等待 collector | 29.43 | CPU/IPC/流水线等待 |
| collector 等待 action | 17.89 | learner 服务推理的等待 |
| collector replay 写入 | 17.98 | 包含共享流水线相关等待 |
| learner inference | 1.45 | 相对较小 |
| replay H2D submit | 13.09 | 提交线程 span，不能等同于纯 DMA 时间 |

这些指标不能直接相加。它们说明双卡额外成本并非仅来自更大的矩阵乘法，而是在通信与流水线同步处表现出来。

### 6.3 每轮第一个 collective 是重点

`diagnostic_v3_retry` 使用独立 event 记录每一次 collective，尾段均值如下：

| collective | rank 0 ms | rank 1 ms |
|---|---:|---:|
| 第 1 次 critic | 22.803 | 25.454 |
| 后续 critic（各次范围） | 0.522–0.559 | 0.551 左右 |
| actor（每轮 2 次） | 0.172–0.173 | 0.171 左右 |
| temperature（每轮 8 次） | 0.055–0.057 | 0.015–0.016 |

同一轮中，critic 梯度尺寸基本相同，而第一次远慢于后续七次。这是已测现象。合理推断是第一次 collective 吸收了跨 rank 到达时差、图或流调度、NCCL 启动/进展等待等成本；目前不能仅凭 event 定位其中每个因素的精确占比。两 rank 都慢也提示不能简单解释成“只有快 rank 等慢 rank”。第 6.5 节的独立 Nsight 运行进一步确认了该运行中确实存在跨 rank 到达时差，但不能把其受追踪扰动的时间直接套用到这里。

由此，下一步优化应优先针对整轮开始处的调度与流水线，而不是盲目减少网络大小或把所有 18 次通信都按首轮耗时估算。

### 6.4 隔离通信 microbenchmark 的用途与限制

使用接近 actor、critic 网络状态尺寸的 payload，GPU 0–1 两 rank 隔离测试中，rank 0 三次试验的中位 CUDA event 时间如下（微秒）。raw 路径包括 all-reduce 及平均 scale kernel：

| payload bytes | 近似对应 | SHM raw | 默认 NET raw | NET / SHM |
|---:|---|---:|---:|---:|
| 4 | temperature 标量 | 52.21 | 56.32 | 1.08× |
| 896,976 | actor 状态尺寸 | 130.18 | 285.79 | 2.20× |
| 3,936,444 | critic 状态尺寸 | 487.95 | 1,225.24 | 2.51× |

包含 pack/unpack 的 DP 路径在上述两个网络尺寸下，SHM 中位数分别约 549.72 / 558.05 μs，NET 约 551.84 / 1,229.50 μs。隔离测试确认较大 payload 的 SHM 路径更快；但 actor 大小的 pack/unpack 路径差异很小，说明提交与打包成本也会影响结果。

原始记录为 `comm_exact_shm.json` 和 `comm_exact_net.json`。这些尺寸依据网络状态字节数选取，不应理解为已经精确扣除了所有 buffer、并包含 finite sentinel 的实际梯度 payload。

作为补充，早期五种尺寸的 GPU 0–1 SHM 测量如下，仍为 rank 0 中位 CUDA event 时间（微秒）：

| payload bytes | raw all-reduce | 带 pack/unpack 的 DP 路径 |
|---:|---:|---:|
| 4 | 59.3 | 117.7 |
| 262,144 | 58.3 | 563.1 |
| 1,048,576 | 149.6 | 589.9 |
| 8,388,608 | 997.0 | 1,067.7 |
| 16,777,216 | 1,957.8 | 2,028.6 |

这些是隔离 eager 提交循环的测量，包含与图内运行不同的调度开销及 NCCL stream dependency，不能与训练中 event 逐项相减得出“计算耗时”。每次试验连续测量 100 次调用，正式计量前预热 20 次；墙钟测量还包括结尾同步。

### 6.5 Nsight Systems：确认等待现象，区分追踪扰动

另进行 210 iterations 的独立真实训练，在第 190–200 轮附近打开 Nsight Systems 捕获窗口，使用 `--cuda-graph-trace=node` 展开图内 kernel。导出的记录包含 20 次 `cudaGraphLaunch` 和 360 个 float32 collective kernel，符合两 rank 各 10 次 replay、每次 18 个梯度 collective。此诊断运行不纳入任何吞吐量均值。

以 device 和 correlation ID 对图 replay 分组，排除首组后观察到：

| 观测 | 追踪中的耗时 |
|---|---:|
| GPU 0 每轮首个 collective | 约 0.31–0.32 ms |
| GPU 0 后续 17 个 collective 合计 | 约 4.01–4.07 ms |
| GPU 1 每轮首个 collective | 约 69.57–116.19 ms |

一个具体样本中，rank 1 的首 collective 从 trace 相对时间 432.303 ms 开始，至约 505.002 ms 结束；rank 0 对应 collective 到 504.647 ms 才开始，持续约 0.313 ms。两者结束时刻接近，而开始时间相差约 72 ms。这直接支持：**该追踪运行的长 NCCL kernel 包含等待另一 rank 到达的时间，不能全部归因为搬运梯度所需的带宽时间。**

但 Nsight node tracing 明显改变了 CPU 与 launch 时序。此次 trace 中 `cudaGraphLaunch` API 平均约 15.19 ms、最大约 61.37 ms，`cudaMemcpyAsync` API 平均约 5.71 ms、最大约 124.98 ms。这些是插桩运行的 API 时长，不是正常训练的基线；也不能据此断言某一个 CPU 锁、线程调度或 NCCL 内部机制已经被唯一定位。

因此，本报告以低扰动 CUDA event 测得的 22–25 ms 首 collective 区间作为正常诊断定量依据，以 Nsight 提供“跨 rank 到达时差可被 NCCL kernel 吸收”的机制证据。两种证据相互补充，不能混算为一个新的性能结果。

完整追踪保存在服务器 `nsys_steady.nsys-rep` 和 `nsys_steady.sqlite`；汇总见 `nsys_stats.log` 与 `nsys_analysis.json`。后者也已同步到本地 `results/`。

### 6.6 线程池对照解释了最大的可消除开销

仅改变仿真 worker 的 CPU 列表后，诊断运行的梯度同步 event 总和从约 30.21 ms 降至 **13.03 ms/rank**，聚合 learner 时间降至 **24.32 ms**。同一次 rank-0 trace 中，collector replay 写入从约 18.54 ms 降至 **1.09 ms**，等待 action 从约 16.39 ms 降至 **3.17 ms**，learner 等待数据从约 30.85 ms 降至 **14.65 ms**；这些数来自独立诊断，不能相加或替代正式吞吐样本。

因此有受控证据表明 CPU 线程池配置影响了 learner 进展、IPC 与同步等待，而不是只有物理仿真速度。具体是 OS 调度、共享内存争抢还是库内部线程池行为各占多少，尚未单独分离。没有依据仅凭长 NCCL kernel 就判定 PCIe 带宽不足。

MuJoCo 使用 CPU 推进物理；增加 GPU 并没有增加服务器总 CPU 资源。最终配置是已有公开 CPU 池接口的本机调优，没有将这台服务器的 CPU 编号硬编码到 `uni_rl` 默认策略中。

## 7. 正确性与质量验证

### 7.1 已完成检查

| 检查 | 结果与范围 |
|---|---|
| 完整 CPU 测试 | 388 passed，45 skipped，3 deselected；覆盖率 69% |
| Ruff | 通过代码与相关测试检查 |
| 格式 | 78 个文件通过格式检查 |
| mypy | 74 个源码文件，无问题 |
| Pyright | 0 errors，0 warnings |
| 真实双 rank 图更新一致性 | 通过；模型与 optimizer 状态跨 rank 最大误差为 0 |
| warmup 状态 / RNG | 已有双 rank probe 验证保持不变 |
| 3 次 replay 更新计数 | critic 24、temperature 24、actor 6，符合预期 |
| 增强双 rank probe，无 loss compile | `enhanced_correctness.json`：通过，无 spawn error |
| 增强双 rank probe，loss compile 开启 | `enhanced_compiled.json`：通过，无 spawn error |
| 真实 CUDA 单元测试 | `tests/algos/test_fast_sac_compile.py`：32 passed，136.63 s |
| 正式重复训练 | 6 次训练均成功完成并退出 |

合成正确性 probe 每轮比较 212 个状态 tensor，检查 actor、critic、target critic 和 temperature 确实更新。它不证明训练收敛，也不证明所有异常输入场景。

第一次完整 CPU 测试曾有一个旧 metric schema fixture 失败：测试通过 `object.__new__` 构造 runner，却未提供新增统计分支所需的 learner / trace 字段。修复测试 fixture 并补充 timing 开关组合后，完整套件通过；没有用删掉测试的方式处理。

### 7.2 新增/扩展的保护

- 梯度图正常退出及 IPC close 异常后的销毁顺序。
- capture / replay / recapture 的计数和 event 生命周期。
- 关闭 event 计量后仍准确输出调用次数，并省略毫秒指标。
- finite sentinel 的同步与门控逻辑。
- batch shape、每轮更新次数改变时重新捕获的正确性。

增强 GPU probe 已在 loss compile 关闭和开启两种模式下通过；两种模式都实际使用双 rank NCCL 和 CUDA Graph。每次 replay 记录 18 次同步及正的 event 时长；随后把 batch 从 64 改为 65、每轮 critic 更新从 8 改为 4，重新捕获后同步计数为 9，warmup 不改变状态及 RNG，212 个状态 tensor 的跨 rank 最大误差仍为 0。

probe 还分别对 critic、temperature、actor 优化器注入 rank 0 非有限 loss、保持梯度有限的情况。三个检查均确认两 rank 同时跳过更新，模型及 optimizer 状态不变、target 不变，跨 rank 最大误差为 0。该测试证明这三个构造场景的门控一致性；不将它外推为所有数值异常均已覆盖。

FastSAC 编译相关单元测试已在真实 CUDA 可见条件下全部通过。Nsight 诊断已完成，其观测及追踪扰动限制见第 6.5 节。

## 8. 修改文件与复现

### 8.1 源码范围

服务器仓库 `unilab_rl` 内：

| 文件 | 主要改动 |
|---|---|
| `src/uni_rl/algos/fast_sac/learner.py` | 允许图内 DP；图生命周期钩子；跨 rank finite sentinel |
| `src/uni_rl/ipc/dp_sync.py` | 图内 collective 计数、可选 event、统计 payload 批量提交 |
| `src/uni_rl/offpolicy/double_buffer_runner.py` | 连接生命周期、指标开关、先释放图再关闭通信 |
| `tests/algos/test_fast_sac_compile.py` | 编译/梯度同步相关回归 |
| `tests/algos/test_dp_graph_lifecycle.py` | 新增图与 process group 生命周期测试 |
| `tests/ipc/test_dp_sync.py` | 图计量及同步逻辑测试 |
| `tests/logging/test_metric_schema.py` | 明确 per-rank 单位和 event 关闭语义 |

### 8.2 重复训练命令

在服务器上使用与本报告相同的三个仓库基线及补丁后：

```bash
cd /mnt/gfs/home/liangyu/desktop/UniLabSim/UniLab
export PYTHONPATH="$PWD/src:$PWD/../unilab_rl/src:$PWD/../unisim/src"

# 推荐本机配置：GPU 0–1，SHM，每 rank 32 个仿真物理核。
# 不要在外层加 taskset 或 numactl；输出目录必须尚不存在。
uv run --no-sync python ../experiments/sac_20261002/run_optimized.py \
  --cards 2 --iterations 1000 \
  --log-dir /mnt/gfs/home/liangyu/desktop/UniLabSim/experiments/sac_20261002/reproduce_optimized_dual

# 同样每 rank 32 核的单卡基线。
uv run --no-sync python ../experiments/sac_20261002/run_optimized.py \
  --cards 1 --single-pool local32 --iterations 1000 \
  --log-dir /mnt/gfs/home/liangyu/desktop/UniLabSim/experiments/sac_20261002/reproduce_optimized_single

# 复现中间阶段的自动 CPU 配置（约1.715×），不是最终推荐配置。
# 自动交替单卡/双卡，共三组；prefix 必须取一个未使用的新名字。
uv run --no-sync python ../experiments/sac_20261002/run_matrix.py \
  n1 n2_shm --iterations 1000 --repeats 3 --prefix reproduce_01
```

`run_optimized.py` 显式选择 GPU 0–1、启用 SHM、恢复默认 NCCL 算法/协议与 graph 调度策略。CPU 编号是此服务器的已测配置，迁移机器需按拓扑重新验证。库内自动 CPU 分配策略没有被硬编码修改，因此要得到最终性能必须应用上述线程池配置。

下面是未覆盖 CPU 池的基础 GPU 0–1 训练命令，便于对照：

```bash
NCCL_SHM_DISABLE=0 uv run --no-sync python \
  src/unilab/scripts/train_sac.py \
  task=g1_walk_flat/mujoco \
  'training.devices=[0,1]' \
  training.no_play=true \
  training.trace_enabled=false \
  algo.max_iterations=1000 \
  training.log_dir=/mnt/gfs/home/liangyu/desktop/UniLabSim/experiments/sac_20261002/reproduce_single_run
```

对照使用 `[4,5]` 或 `[0,4]`。Hydra 列表参数需要引号，尤其在服务器 zsh 下。不要复用已经存在的输出目录，也不要在同一卡上并发跑多个性能作业。

### 8.3 诊断与分析命令

```bash
# 独立诊断运行启用 trace；不可与正式无 trace 样本混为同一组。
uv run --no-sync python ../experiments/sac_20261002/run_matrix.py \
  n2_shm --iterations 1000 --trace --prefix reproduce_trace

# 重算一个已有运行的统计。
uv run --no-sync python ../experiments/sac_20261002/analyze_run.py \
  /mnt/gfs/home/liangyu/desktop/UniLabSim/experiments/sac_20261002/reproduce_single_run \
  --output /mnt/gfs/home/liangyu/desktop/UniLabSim/experiments/sac_20261002/reproduce_single_run_metrics.json

# 小模型的真实 NCCL + CUDA Graph 正确性；不是吞吐量 benchmark。
CUDA_VISIBLE_DEVICES=0,1 NCCL_SHM_DISABLE=0 uv run --no-sync python \
  ../experiments/sac_20261002/dp_probe.py correctness \
  --output /mnt/gfs/home/liangyu/desktop/UniLabSim/experiments/sac_20261002/reproduce_correctness.json
```

## 9. 原始证据索引与失败实验

服务器证据根目录：`/mnt/gfs/home/liangyu/desktop/UniLabSim/experiments/sac_20261002/`。本地结果副本位于本报告相邻的 `results/`，脚本位于本报告同目录；完整 trace、TensorBoard 和 checkpoint 以服务器各 run 目录为准。

最终交付的 `unilab_rl_optimization.patch` 包含 3 个生产文件和 4 个测试文件（包含新增文件），已在当前修改树通过 `git apply --reverse --check` 验证。`sac_experiment_evidence.zip` 收录源码快照、完整补丁、实验脚本、日志、指标、TensorBoard、选定 Perfetto trace 和 Nsight report；`evidence_manifest.json` 提供文件 SHA-256。大型 checkpoint 与 Nsight SQLite 保留在服务器，未打入压缩包。

| 证据 | 内容 |
|---|---|
| `baseline.json`、`baseline_n1_metrics.json` | 原始单卡与双卡失败基线 |
| `final_r{1,2,3}_{n1,n2_shm}_execution.json` | 正式重复实验命令、SHA、退出状态和主指标 |
| `pool_final_r{1,2,3}_{n1,n2_shm}_execution.json` | 最终线程池配置的交替重复实验 |
| `local32_r{1,2,3}_n1_execution.json` | 单卡本地 32 核的三次保守基线 |
| 对应 `*_metrics.json` | 尾段统计、配置、summary |
| 对应 run 目录 | TensorBoard、run_config、run_summary、checkpoint |
| 每次 `*.patch` | 当次受 Git 跟踪源码的 diff 快照 |
| `diagnostic_v3_retry/collective_events_rank*.json` | 每次 collective 的独立 event 记录 |
| `comm_01_shm.json`、`comm_exact_*.json` | 隔离通信结果 |
| `final_correctness.json` | 已完成的双 rank 一致性检查 |
| `enhanced_correctness.json`、`enhanced_compiled.json` | 增强 probe，两种 loss compile 模式均通过 |
| `nomix_clean_n2_shm_*`、`graph_order_n2_shm_*`、`numa0_n2_shm_*` | 图调度与 NUMA 单次 trace 对照 |
| `nsys_steady.nsys-rep`、`nsys_steady.sqlite` | 独立 Nsight 图节点追踪及数据库 |
| `nsys_stats.log`、`nsys_analysis.json` | Nsight 汇总、collective 时间线和 API 统计 |
| `final_*tests.log`、类型检查日志 | 测试和静态检查输出 |

每次运行的 `git diff` 快照不自动包含未跟踪的新文件；最终完整补丁已额外包含新增的生命周期测试。

以下失败保留用于说明实验过程，不计入成功样本：

1. 原始双卡默认训练：whole-cycle DP 互斥检查抛错。
2. `graph_socket`：使用相对 `file://` rendezvous 路径导致启动失败；改为绝对路径。
3. `graph_socket_abs` 和 `transport_n2_shm`：训练完成但 teardown 挂起；修复图销毁顺序后重跑。
4. 最初 `diagnostic_v3`：诊断 harness 缺少 `if __name__ == "__main__"`，spawn collector 重入训练；修复入口保护后使用 `diagnostic_v3_retry`。失败运行中的 connection/cubin 警告不用于性能结论。

## 10. 未验证项及后续优化优先级

目前已确认双卡路径可保留整轮编译图并正确同步。仅代码与 SHM 配置在自动 CPU 布局下约为 1.715×；进一步控制仿真池后，双卡约 108.6k steps/s，相对本地 32 核单卡约为 **2.165×**，在本机这组稳态短程实测中达到目标。加速比必须同时注明单卡 CPU 池布局，不能将联合系统优化称为纯 GPU 线性扩展。

图混合调度、整进程 NUMA / CPU 绑定和协议切换已经完成对照，未选为推荐。下一阶段若继续优化，应针对剩余跨 rank 到达时差和 CPU/IPC 进展做低扰动剖析，并用更长训练和多个 seed 验证收益；本次没有改变同步频率或 SAC 更新语义。

当前没有验证跨多种 seed 的长期性能、收敛到相同 reward 的耗时、其他任务/后端、4 卡或 8 卡扩展。任何进一步改变 batch、更新频率、日志频率、同步频率的实验都应单独标注，不能混入本报告的默认参数加速结论。
