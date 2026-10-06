# 来源、完整性与许可

本仓库由 2026-10-02～03 在 `5090-server2` 上完成的实测资料整理而来。2026-10-06 的工作是文档、目录与便携脚本整理，没有重新生成性能数字。

- `patches/unilab_rl_optimization.patch` 与原交付补丁字节一致，目标commit见README。
- `results/2026-10-03/raw/` 来自原始JSON，保留运行命令、历史路径、host信息和配置作为实验溯源。
- `artifacts/sac_20261002_evidence.zip` 为原始归档，SHA-256：`76d0121e4456d441b658c4f3568c8d2af811eaaf631cee543f5fff1de3e3bdff`。
- `artifacts/evidence_manifest.json` 记录原归档335个资料项的逐文件SHA-256。归档本身还包含该manifest；manifest不对自身做递归哈希。
- 归档中的原始报告与脚本保留历史路径，适合审计；当前 `scripts/` 支持 `UNILAB_WORKSPACE`、输出目录和dry-run，适合新路径复现。详细报告顶部也注明当前入口。
- checkpoint和完整Nsight SQLite未打包，所需训练配置、统计依据、主要trace和源补丁已包含。

上游项目为 [UniLab](https://github.com/unilabsim/UniLab)、[unilab_rl](https://github.com/unilabsim/unilab_rl)、[unisim](https://github.com/unilabsim/unisim)。补丁与归档包含修改后的 `unilab_rl` 源码/测试，原有权利声明继续适用；随附其 [Apache-2.0许可证](../licenses/unilab_rl-LICENSE)。本说明不替其他上游项目或第三方资产重新授权。

使用 `scripts/verify_repository.py` 可校验归档和补丁完整性、Python语法、主要结果与原始execution文件的均值一致性。不需要GPU。
