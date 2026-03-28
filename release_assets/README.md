# Release Assets

这个目录保存与论文可复现性直接相关、且适合公开发布的轻量资产。

当前包含：

- `agiqa3k_split_seed42.json`
  - 主稿 AGIQA-3K 固定划分文件。
  - 用于主线单次重跑或对齐 `split_seed=42` 的实验协议。

- `logic_feature_names.json`
  - 12 维 logic interface 的固定顺序。
  - 与 `vectorize_direct_alignment_features.py` 和 `src/funnel.py` 保持一致。

- `logic_cache_schema.md`
  - Stage A audit 输出与 12 维压缩向量的字段说明。
  - 说明哪些字段是 requirement-level cache，哪些字段直接进入在线残差校准。

注意：

- 仓库不分发 AGIQA-3K 和 AIGCIQA2023 原始图像与标注。
- 与 benchmark 再分发条款相关的派生预测文件和缓存，不在这个公开 release 中直接打包；如有需要，可按论文中的声明联系作者获取。
