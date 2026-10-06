# 仓库整理验证（2026-10-06）

本次发布没有重新运行 GPU 性能实验，也没有改动算法补丁。完成的独立检查：

- 原证据包整体 SHA-256、清单中 335 个文件长度与 SHA-256 全部匹配。
- 发布的补丁与原证据包内补丁字节一致。
- 从 15 个成功运行的 execution JSON 重算五组均值，与汇总一致；本地32核基线加速比为 `2.1645177154253528`。
- 对全部当前 Python 脚本做 AST 语法检查；Markdown 本地链接均可解析。
- 将上游 `e2f18b4df1dcb0c35cd4c2dccd0517151633e53d` 用 `git archive` 导出到独立目录，`git apply --check` 通过，未改动现有工作树。
- `run_optimized.py --dry-run` 和三组 `run_matrix.py ... --dry-run` 可正确展开工作区、输出目录、GPU及CPU池参数；dry-run无需Torch/TensorBoard或GPU。

历史训练、CUDA/CPU测试与静态检查记录见 [validation](../results/2026-10-03/validation/)；本次资料校验可运行 `scripts/verify_repository.py` 重复。
