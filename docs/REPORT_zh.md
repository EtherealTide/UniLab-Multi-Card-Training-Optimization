# UniLab 双卡 SAC 优化技术报告

实测：2026-10-02～03；整理：2026-10-06。任务为真实 `g1_walk_flat/mujoco` SAC 训练，目标是在保持每 rank 默认训练参数的前提下，降低 learner 耗时并提高总环境步吞吐。

**最终结果：单卡 50,194.19、双卡 108,646.21 env steps/s，均值比 2.165×。** 这来自图内梯度同步、SHM 与仿真 CPU 池配置的组合；属于 CPU / collector / GPU 一起扩展的端到端弱扩展，不能解释为补丁独立提升 2.165×，也不代表收敛加速。

## 1. 初始问题与最终设计

初始单卡已经使用 Inductor + 整轮 CUDA Graph；双卡会触发 `FastSAC NVIDIA CUDA whole-cycle mode does not support DP fallback`，因此没有有效的原始默认双卡吞吐。短单卡基线 learner 约 11.21 ms，环境 step 约 60.80 ms，说明仅减少 GPU 计算不足以解决整体吞吐问题。

| 层 / 改动 | 初始问题 | 最终实现及理由 |
|---|---|---|
| learner：整轮图内 DP | CUDA Graph 与梯度同步被互斥检查阻止 | 移除限制，在每个 optimizer step 前捕获 pack → all-reduce → average → unpack；保留编译、整轮 replay 和每轮 18 次 collective，不减少更新次数 |
| learner / IPC：图生命周期 | 捕获、重捕获与回调状态缺少明确衔接 | 增加 `begin_capture / end_capture / before_replay` 钩子；更换同步回调时使旧图失效，重捕获时重新统计，避免继续使用旧通信状态 |
| learner：跨 rank 有限性 | 单 rank 非有限 loss 即使梯度有限，也必须让所有 rank 一致跳步 | 将 float32 finite-loss sentinel 随梯度归约，结合梯度有限性检查统一门控 critic、temperature、actor 与目标更新；不增加第 19 次 collective，sentinel 不进入参数或 checkpoint |
| runner：资源释放 | 早期训练结束后退出挂起 | 先解绑同步回调并释放图，再关闭同步器 / process group；IPC 清理异常也通过 `finally` 保证释放顺序 |
| IPC / runner：统计 | capture 的 Python 调用不等于 replay 次数；细碎标量提交增加开销 | 捕获期计数与 replay 计数分离；仅诊断开启 external CUDA Event，未计时不伪报 0 ms；统计 payload 在 host 合并后一次 H2D |
| 部署配置：SHM | 本机 P2P 不可用，默认禁用 SHM 时实际走 NET/IB | 启用 `NCCL_SHM_DISABLE=0`，日志确认 SHM/direct；减小本机传输开销 |
| 部署配置：CPU 池 | 自动线程池与整个进程树绑核均不理想 | 只给 MuJoCo 仿真池分配物理核，learner/NCCL 保持原有亲和性；改善 IPC 与跨 rank 进度等待 |

架构边界保持不变：learner 管优化与图生命周期，IPC 管归约与计量，runner 管连接和销毁；环境仍由 `EnvFactory` 注入，`uni_rl` 不导入 UniLab / UniSim。CPU 编号通过现有 UniLab 配置传入，未写入算法库默认策略；没有改动仿真 ABI、任务语义或 cold/hot path 边界。有限性门控主要针对本次 CUDA / 无 GradScaler 路径，不能替代其他设备和混合精度路径的专项验证。

补丁包含 3 个生产文件：`algos/fast_sac/learner.py`、`ipc/dp_sync.py`、`offpolicy/double_buffer_runner.py`（均位于 `src/uni_rl/`）；另有 4 个测试文件，覆盖 compile、图生命周期、同步和 metric schema。没有逐项独立消融，不能为统计合并等小改动分配单独收益百分比。

## 2. 实验版本、硬件与参数

最终实验代码 = 下列三个固定版本 + [完整 RL 补丁](../patches/unilab_rl_optimization.patch)。UniLab 与 UniSim 没有本次生产代码改动。补丁以原始字节归档；不是仅凭分支名称定位代码。

| 组件 | 版本 |
|---|---|
| UniLab | `0fd2bd5d72210ad685838ade4ba4687b269f7de2` |
| unilab_rl 1.4.4 | `e2f18b4df1dcb0c35cd4c2dccd0517151633e53d` + 本仓库补丁 |
| UniSim | `783220d64ba7a207d316f1dd06ccd1a1db1261bc` |
| PyTorch / CUDA / NCCL | `2.14.0+cu130` / 13.0 / 2.30.7 |
| NVIDIA driver / MuJoCo | 580.95.05 / 3.11.0 |
| 服务器 | 8×RTX 5090（每卡 32607 MiB）；2×Xeon Platinum 8380，80 物理核 / 160 线程；约 1 TB RAM |

GPU 0–1、4–5 为 PIX，0–4 为跨 NUMA SYS；0–1 peer access 实测不可用。NUMA 0 CPU 为 `0–39,80–119`，NUMA 1 为 `40–79,120–159`。最终用 GPU 0–1；两仿真池分别为物理核 `8–39`、`48–79`，不使用对应 SMT 兄弟线程。第二个 CPU 池位于 NUMA 1，并不与 GPU 1 所在 NUMA 相同；本方案是实测配置，不是通用 GPU/CPU 同节点配对规则。

| 参数 | 单卡 | 双卡 |
|---|---|---|
| task / 算法 | `g1_walk_flat/mujoco` / SAC（FastSAC） | 相同 |
| 环境数 | 2048 | 每 rank 2048，全局 4096 |
| batch size | 8192 | 每 rank 8192，全局 16384 |
| 每轮更新 | critic 8、temperature 8、actor 2 | 每 rank 相同；合计 18 次梯度 collective / 轮 |
| 更新频率 | `updates_per_step=8`、`policy_frequency=4`、target frequency 1 | 相同 |
| 编译 / 精度 | Inductor + 整轮 CUDA Graph；AMP auto | 相同 |
| 其他 | seed 1；replay prefetch `one_tick`；`env_steps_per_sync=1`；TensorBoard interval 1 | 相同 |
| 正式测量 | 1000 iterations；trace 关闭 | 相同 |
| 推荐仿真池 | 本地 32 物理核 | 每 rank 32，共 64 核 |

### 指标与可比性

主吞吐 = 最后 50% TensorBoard 样本中 **累计总环境步数差 / wall-time 差**；双卡使用 rank 0 记录的全局累计步数。包含测量窗口内日志等开销，排除启动和首次编译。不是 replay rows/s，也不是 `mean(Perf/total_fps)`；后者在自动 CPU 双卡组给出 62,342～69,423，明显高于共同墙钟口径。

learner 列是尾段记录的 learner 耗时均值，用于定位，不用于反推主吞吐。异步环境、计算与等待会重叠，各阶段不能直接相加。三次重复均使用 seed 1，是运行重复而非多 seed 收敛实验。

## 3. 正式实验数据

以下 15 次均为 1000 iterations、trace 关闭、退出码 0。自动 CPU 的单/双卡三组交替运行；64 核单池与双池三组交替运行；本地 32 核单卡是随后连续补充的三次，不是与最终双卡交错配对。

| 配置 | 第 1 次 steps/s | 第 2 次 | 第 3 次 | 均值 | 样本标准差 |
|---|---:|---:|---:|---:|---:|
| 自动 CPU 单卡 | 31,008.18 | 30,708.93 | 30,580.36 | 30,765.82 | 219.51 |
| 自动 CPU 双卡 + SHM | 54,089.93 | 54,105.31 | 50,054.20 | 52,749.81 | 2,334.49 |
| 单卡跨 NUMA 64 核池 | 42,506.53 | 46,352.36 | 47,670.96 | 45,509.95 | 2,683.29 |
| 单卡本地 32 核池 | 48,057.33 | 51,036.37 | 51,488.87 | **50,194.19** | 1,864.36 |
| 双卡各 32 核池 + SHM | 107,503.16 | 109,619.53 | 108,815.94 | **108,646.21** | 1,068.35 |

| 配置 | 第 1 次 learner ms | 第 2 次 | 第 3 次 | 均值 ms |
|---|---:|---:|---:|---:|
| 自动 CPU 单卡 | 11.079 | 11.382 | 11.334 | 11.265 |
| 自动 CPU 双卡 | 41.487 | 41.531 | 45.367 | 42.795 |
| 单卡 64 核池 | 13.268 | 11.878 | 12.206 | 12.451 |
| 单卡本地 32 核池 | 12.994 | 13.141 | 13.255 | 13.130 |
| 双卡各 32 核池 | 21.650 | 24.423 | 24.581 | **23.552** |

最终双卡相对自动 CPU 双卡吞吐约 **2.060×**，learner 均值由 42.795 降至 23.552 ms。最终单/双卡本地池均值比为 **2.165×**；总仿真核数从 32 增至 64。若与跨 NUMA 的单卡 64 核池比为 2.387×，但池的 NUMA 布局、collector 数量仍不同，不能据此推导纯 GPU 超线性扩展。

原始数据位于 [raw](../results/2026-10-03/raw/)：自动组 `final_r{1,2,3}_{n1,n2_shm}_execution.json`；64 核 / 双池组 `pool_final_r{1,2,3}_{n1,n2_shm}_execution.json`；本地单卡组 `local32_r{1,2,3}_n1_execution.json`。对应 `_metrics.json` 保存阶段指标；[summary.json](../results/2026-10-03/summary.json) 保存汇总。

## 4. 为什么选择这些优化

### 通信不是全部瓶颈

隔离 eager all-reduce 的 rank 0 中位数如下。payload 来自模型状态的近似尺寸，实际梯度与 sentinel 尺寸略有不同；这些值不能直接替代训练图内耗时。

| 近似 payload | 默认 NET/IB μs | SHM μs |
|---|---:|---:|
| 4 B | 56.32 | 52.21 |
| 896,976 B | 285.79 | 130.18 |
| 3,936,444 B | 1,225.24 | 487.95 |

真实训练 CUDA Event 中，首个 critic 同步 rank 0 / 1 约 **22.803 / 25.454 ms**，后续相近尺寸同步约 **0.522～0.559 ms**，actor 约 0.172 ms。首个长区间包含 pack、依赖与等待，不能全部解释为传输。

Nsight 在约第 190～200 轮记录到 20 次 graph launch、360 次 float32 collective，与每图 18 次一致。一个样本中 GPU 1 的 collective 约 432.303 ms 开始、505.002 ms 结束，另一 rank 约 504.647 ms 才到达；支持“到达等待”解释。逐节点追踪明显扰动运行，这些绝对时间不用于正式性能统计。

仅调整仿真池后的独立诊断结果：

| 诊断指标 | 自动池 ms | 每 rank 32 核池 ms |
|---|---:|---:|
| 梯度同步 event 总和 / rank | 30.21 | 13.03 |
| 聚合 learner 更新 | 41.65 | 24.32 |
| rank 0 collector replay 写入 | 18.54 | 1.09 |
| rank 0 collector 等待 action | 16.39 | 3.17 |
| rank 0 learner 等待数据 | 30.85 | 14.65 |

CPU 池布局影响了 learner 进度与 IPC，而不只是物理仿真速度。操作系统调度、共享内存竞争和线程池内部行为各占多少尚未隔离；这些重叠阶段不能求和或替代正式样本。

### 探索与负面结果

下表保留影响决策的对照；中间代码和诊断设置并不完全相同，不是严格单因素消融。200 轮挂起案例仅是已记录的训练段，不算完成的性能样本；其余表内探索各为单次运行。

| 运行 / 配置 | iterations | steps/s | learner ms | 说明 |
|---|---:|---:|---:|---|
| 原始单卡 | 200 | 约 29,940 | 11.210 | 短基线 |
| 图内 DP，NET/IB | 200 | 44,059.11 | 56.999 | 退出挂起 |
| 图内 DP，SHM | 200 | 49,681.69 | 42.500 | 退出挂起 |
| SHM + 生命周期修复 | 200 | 46,465.29 | 47.254 | 正常退出 |
| 早期无 trace 单卡 / 双卡 | 1000 | 31,117.68 / 57,031.46 | 11.299 / 38.197 | 单次 1.833×，不作最终结论 |
| 中间版 GPU 0–1 | 1000 | 53,914.14 | 41.645 | 含 event 计量 |
| 中间版 GPU 4–5 | 1000 | 53,876.87 | 41.513 | PIX 对照 |
| 中间版 GPU 0–4 | 1000 | 49,592.96 | 45.093 | SYS，比 0–1 低约 8% |
| 禁用 graph mixing | 1000 | 54,472.57 | 40.654 | trace 开启，无重复收益证据 |
| 再禁用 graph stream ordering | 1000 | 45,089.22 | 48.558 | trace 开启，未采用 |
| 整个进程树绑 NUMA 0 | 1000 | 42,229.65 | 52.651 | trace 开启，未采用 |
| 整个进程树绑 64 物理核，单卡 / 双卡 | 1000 | 30,036.85 / 44,940.14 | 11.040 / 50.585 | trace 开启，未采用 |
| 仅仿真池各 32 核 | 1000 | 109,565.41 | 24.324 | trace 开启，进入正式复测 |
| 强制 Simple 协议，自动池 | 1000 | 53,523.60 | 42.219 | trace 开启，未见收益 |

因此保留 NCCL 默认算法、协议与 graph 调度设置，只启用本机已验证的 SHM。早期受并发 CPU 测试干扰的 `nomix` 运行不用于结论；表内为清理干扰后的复测。

## 5. 正确性与适用边界

[验证日志](../results/2026-10-03/validation/)记录：CPU 套件 **388 passed / 45 skipped / 3 deselected**，覆盖率 69%；CUDA compile 测试 **32 passed**；mypy 74 文件无问题，Pyright 零错误 / 警告，Ruff 检查通过。

真实双 rank NCCL probe 同时覆盖 eager loss 和 Inductor：212 个状态张量完全一致（最大误差 0）；warmup 不改变状态 / RNG；3 次 replay 对应 24 次 critic、24 次 temperature、6 次 actor 更新；batch 64→65、更新数 8→4 的重捕获使 collective 数从 18→9；单 rank Inf loss、梯度仍有限时，各 rank 的三种 optimizer 与目标更新一致跳过。检查范围包含参数和 optimizer 状态，不只是 loss 日志。

尚未验证多 seed 长期收敛、固定全局 batch 的强扩展、4/8 卡、其他后端或所有软硬件版本。全局 batch 加倍可能改变学习轨迹；吞吐提高不等于到达同等奖励的时间同比降低。三次样本不构成长期性能保证。本次文档整理未重跑 GPU 实验。

## 6. 复现入口与证据

真实训练和三次矩阵复测见 [README](../README.md)。下面使用其复测章节设置的路径与 `PYTHONPATH`，从 UniLab 目录执行诊断；输出目录须为新目录。性能测试与诊断分开串行运行。

```bash
# 隔离通信；改成 NCCL_SHM_DISABLE=1 可对照默认传输。
CUDA_VISIBLE_DEVICES=0,1 NCCL_SHM_DISABLE=0 uv run --no-sync python \
  "$OPT_REPO/scripts/dp_probe.py" communication \
  --bytes 4 896976 3936444 --warmup 20 --iterations 100 --trials 3 \
  --output "$UNILAB_EXPERIMENT_DIR/comm_shm.json"

# 逐 collective CUDA Event；此例为自动CPU池。
CUDA_VISIBLE_DEVICES=0,1 NCCL_SHM_DISABLE=0 uv run --no-sync python \
  "$OPT_REPO/scripts/profile_training.py" \
  task=g1_walk_flat/mujoco 'training.devices=[0,1]' \
  training.no_play=true training.trace_enabled=true algo.max_iterations=1000 \
  "training.log_dir=$UNILAB_EXPERIMENT_DIR/events_diag"
```

诊断最终 CPU 池时追加 README 中用 `seq` 生成的 `training.dp_collector_cpu_ids` 参数。Nsight 入口为 [nsys_training.py](../scripts/nsys_training.py)，支持相同 Hydra 参数，在第 190～200 轮采样；可用以下命令启动自动池诊断，需要预装 `nsys`：

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_SHM_DISABLE=0 nsys profile \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --cuda-graph-trace=node -o "$UNILAB_EXPERIMENT_DIR/nsys_steady" \
  uv run --no-sync python "$OPT_REPO/scripts/nsys_training.py" \
  task=g1_walk_flat/mujoco 'training.devices=[0,1]' \
  training.no_play=true training.trace_enabled=false algo.max_iterations=210 \
  "training.log_dir=$UNILAB_EXPERIMENT_DIR/nsys_training"
```

[原始证据包](../artifacts/sac_20261002_evidence.zip)包含 335 项清单资料（TensorBoard、配置、日志、选定 trace、Nsight report、原始脚本与报告），SHA-256 为 `76d0121e4456d441b658c4f3568c8d2af811eaaf631cee543f5fff1de3e3bdff`；未打包 checkpoint 和完整 Nsight SQLite。历史文档封存在归档中供审计，不再作为当前使用说明。当前 `scripts/` 增加路径配置、dry-run 和案例选择，没有改变算法或重写实测数据。矩阵的 tracked diff 快照不包含 untracked 文件，完整补丁包含新增测试。

`verify_repository.py` 无需 GPU，检查归档逐项哈希、补丁原始字节、15 次正式运行与汇总一致性、脚本语法及文档链接。上游源码 / 测试保留其 [Apache-2.0 许可证](../licenses/unilab_rl-LICENSE)；本报告不对其他上游代码或资产重新授权。
