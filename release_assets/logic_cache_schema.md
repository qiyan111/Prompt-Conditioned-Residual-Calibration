# Logic Cache Schema

## 1. Stage A cache fields

主线实现中的 Stage A 先输出 requirement-level 审计结果，再由后处理压缩为固定 12 维向量。公开 release 关心的核心字段包括：

- `main_visual_subject`: 图像主主体文本。
- `subject_match`: 主主体是否与 prompt 核心主体匹配，二值或 `[0,1]` 标量。
- `present_ratio`: 已满足 requirement 的比例。
- `missing_ratio`: 缺失 requirement 的比例。
- `uncertainty_ratio`: 状态不确定 requirement 的比例。
- `off_topic_ratio`: 离题概念比例。
- `attribute_match_rate`: 属性匹配率。
- `scene_match_rate`: 场景匹配率。
- `style_match_rate`: 风格匹配率。
- `relation_match_rate`: 关系匹配率。
- `count_match_rate`: 计数约束匹配率。
- `contradiction_flag`: 是否存在显式矛盾。
- `confidence`: 审计链的保守置信度。

## 2. 12-dimensional online interface

在线残差校准实际消费的固定顺序如下：

1. `subject_match`
2. `present_ratio`
3. `missing_ratio`
4. `off_topic_ratio`
5. `attribute_match_rate`
6. `scene_match_rate`
7. `style_match_rate`
8. `relation_match_rate`
9. `count_match_rate`
10. `confidence`
11. `uncertainty_ratio`
12. `contradiction_flag`

这一定义与以下文件保持一致：

- `vectorize_direct_alignment_features.py`
- `src/funnel.py`
- 论文 `Table 1`

## 3. Split file format

`agiqa3k_split_seed42.json` 使用的字段包括：

- `split_seed`
- `test_size`
- `num_rows`
- `train_ids`
- `val_ids`
- `test_ids`

这些 id 对应训练 CSV 在固定行顺序下的 `_split_row_id` / 行索引语义，用于在主线重跑中保持可比的 train/val/test 划分。
