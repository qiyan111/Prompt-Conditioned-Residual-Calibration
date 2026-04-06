# Logic Cache Schema

## 1. Stage A Cache Fields

The main pipeline first exports requirement-level audit results from Stage A and then compresses them into the fixed 12-dimensional logic vector used by the online scorer. The public release focuses on the following cache fields:

- `main_visual_subject`
- `focus_image_path`
- `focus_mask_path`
- `focus_valid`
- `focus_bbox_xyxy`
- `focus_area_ratio`
- `focus_center_x`
- `focus_center_y`
- `subject_match`
- `present_ratio`
- `missing_ratio`
- `uncertainty_ratio`
- `off_topic_ratio`
- `attribute_match_rate`
- `scene_match_rate`
- `style_match_rate`
- `relation_match_rate`
- `count_match_rate`
- `contradiction_flag`
- `confidence`

## 2. Fixed 12-Dimensional Online Interface

The online residual scorer consumes the following feature order:

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

This definition is kept consistent across:

- `vectorize_direct_alignment_features.py`
- `src/funnel.py`
- `logic_feature_names.json`
- Table 1 of the manuscript

## 3. Split File Format

`agiqa3k_split_seed42.json` contains:

- `split_seed`
- `test_size`
- `num_rows`
- `train_ids`
- `val_ids`
- `test_ids`

These identifiers correspond to the stable row-order semantics of the training CSV and allow fixed-split reruns that remain comparable with the manuscript results.
