# Prompt-Conditioned Residual Calibration

This repository hosts the code and release assets for the residual-calibration experiments only. Manuscript source files are intentionally not included in this public repository.

Repository URL:
`https://github.com/qiyan111/Prompt-Conditioned-Residual-Calibration`

## Contents

- `train.py`: main training entry for the residual-calibration model.
- `run_model_ablations.py`: scripts for the main ablation runs.
- `vectorize_direct_alignment_features.py`: converts Stage A outputs into the fixed 12-dimensional logic vector.
- `verify_images_direct_alignment_features.py`: structured prompt-image auditing for Stage A.
- `run_external_fair_baselines.py`: fair rerun entry for external baselines.
- `external_baseline_adapter.py`: adapters for external baseline implementations.
- `prepare_aigciqa2023_extension.py`: data preparation for the AIGCIQA2023 extension.
- `run_local_vlm_judge.py`: direct VLM-judge baseline script.
- `make_fig3_ablation_bars.py`, `make_fig4_residual_scatter.py`: figure generation scripts.
- `src/`: core model components.
- `release_assets/`: public split definition and logic interface schema.

## Environment

Use Python 3.10 or later.

```bash
pip install -r requirements.txt
```

Core dependencies include `torch`, `transformers`, `peft`, `openai`, `pandas`, `scipy`, `scikit-learn`, `Pillow`, `tqdm`, and `matplotlib`.

## Data

This repository does not redistribute the original `AGIQA-3K` or `AIGCIQA2023` data. Please obtain the datasets from their official distribution channels and point the scripts to local copies with `--data_csv_path` and `--image_base_dir`.

Public release assets are under `release_assets/`:

- `agiqa3k_split_seed42.json`: fixed AGIQA-3K split used by the main rerun.
- `logic_feature_names.json`: fixed order of the 12-dimensional logic interface.
- `logic_cache_schema.md`: field descriptions for the Stage A cache and 12-dimensional vector.

## Mainline Reproduction

Run Stage A and export the 12-dimensional logic vector:

```bash
python verify_images_direct_alignment_features.py \
  --input_csv /path/to/agiqa.csv \
  --image_root /path/to/AGIQA-3K \
  --output_jsonl outputs/alignment_features_direct.jsonl \
  --output_csv outputs/alignment_features_direct_raw.csv \
  --model YOUR_VLM_MODEL \
  --base_url YOUR_API_BASE \
  --api_key YOUR_API_KEY
```

```bash
python vectorize_direct_alignment_features.py \
  --input_jsonl outputs/alignment_features_direct.jsonl \
  --output_jsonl outputs/alignment_logic_12d.jsonl \
  --output_csv outputs/alignment_logic_12d.csv
```

Run the main ablations:

```bash
python run_model_ablations.py \
  --data_csv_path /path/to/agiqa.csv \
  --image_base_dir /path/to/AGIQA-3K \
  --clip_model_name openai/clip-vit-large-patch14 \
  --funnel_cache_jsonl outputs/alignment_logic_12d.jsonl \
  --output_root runs/agiqa_mainline \
  --split_seed 42 \
  --train_seeds 11,22,33
```

For a single fixed split run, call `train.py` directly and point `--split_file` to `release_assets/agiqa3k_split_seed42.json`.

## External Baselines

`run_external_fair_baselines.py` reruns external baselines under the same split protocol. The repository does not vendor third-party baseline code; prepare those repositories separately and provide their paths through `--external_repo_root` or the corresponding `--*_repo_dir` arguments.
