# Neuro-Symbolic Residual Calibration

This repository directly accompanies the manuscript "Neuro-Symbolic Residual Calibration for Fine-Grained Text-to-Image Alignment Assessment" currently submitted to *The Visual Computer*. If you use the code, split definition, cached logic-feature interface, or reproduction assets, please cite the corresponding manuscript and the DOI-bearing archive listed below.


## Permanent Links

- Current GitHub repository URL: `https://github.com/qiyan111/Prompt-Conditioned-Residual-Calibration`
- Zenodo archive: `https://doi.org/<ZENODO_DOI>`
- Code DOI: `https://doi.org/<CODE_DOI>`
- Data and release-assets DOI: `https://doi.org/<DATA_DOI>`
- Manuscript status: submitted to *The Visual Computer*

## What Is Included

This public repository contains the code and release assets required to reproduce the residual-calibration experiments reported in the manuscript. Manuscript source files are intentionally not included in this repository.

The repository includes:

- training and evaluation code for the proposed residual-calibration model
- Stage A auditing and 12-dimensional logic-vector construction
- controlled ablation and baseline-rerun scripts
- AIGCIQA2023 extension preparation utilities
- figure-generation scripts
- public split definitions and logic-interface schema files

## Repository Layout

- `train.py`: main training and evaluation entry for single runs
- `run_model_ablations.py`: controlled ablation runner for the main model family
- `verify_images_direct_alignment_features.py`: Stage A structured prompt-image auditing
- `vectorize_direct_alignment_features.py`: converts Stage A outputs into the fixed 12-dimensional logic vector
- `prepare_aigciqa2023_extension.py`: builds the local AIGCIQA2023 extension CSV and split JSON
- `run_external_fair_baselines.py`: reruns external baselines under a matched local protocol
- `run_local_vlm_judge.py`: direct VLM-judge baseline script
- `make_fig3_ablation_bars.py`: figure-generation script for the AIGCIQA2023 ablation bars
- `make_fig4_residual_scatter.py`: figure-generation script for the AGIQA-3K residual-scatter plot
- `src/funnel.py`: fixed logic-feature names and cache-field definitions used by the online scorer
- `release_assets/`: public split definition and logic-interface schema required for reproduction
- `requirements.txt`: pip dependency list
- `environment.yml`: conda environment file for reproducible setup
- `CITATION.cff`: citation metadata for the repository and manuscript linkage

## Key Algorithmic Components

The manuscript reports a three-stage evaluation pipeline.

1. Stage A: structured prompt-image auditing
   `verify_images_direct_alignment_features.py` produces requirement-level audit outputs from a prompt-image pair. It is designed for an OpenAI-compatible local VLM endpoint and exports both raw audit records and intermediate structured features.
2. Stage A compression: fixed 12-dimensional logic interface
   `vectorize_direct_alignment_features.py` compresses Stage A audit outputs into the fixed 12-dimensional logic vector used online by the scorer. The released feature order is identical to the definitions in `src/funnel.py` and `release_assets/logic_feature_names.json`.
3. Stage B and Stage C: CLIP backbone plus prompt-conditioned residual calibration
   `train.py` and `run_model_ablations.py` train the model family reported in the manuscript. The online scorer combines CLIP image-text features with the cached 12-dimensional logic vector, and predicts residual corrections for quality and alignment rather than replacing the base scorer.

## Environment and Dependencies

Recommended baseline environment:

- Python `3.10`
- Windows or Linux
- NVIDIA GPU recommended for training and large-model inference
- CUDA-enabled PyTorch build recommended for the main training runs

Create the environment with either of the following:

```bash
conda env create -f environment.yml
conda activate pcrc
```

or

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Core Python dependencies are listed in `requirements.txt` and include `torch`, `torchvision`, `transformers`, `peft`, `openai`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `Pillow`, `tqdm`, and `matplotlib`.

For Stage A auditing and the direct VLM judge, you also need access to a local or remote OpenAI-compatible VLM endpoint. The scripts expect `--model`, `--base_url`, and `--api_key`.

## Data Access and Redistribution Policy

This repository does not redistribute the original `AGIQA-3K` or `AIGCIQA2023` images or official raw annotations. Please obtain the datasets from their official distribution channels and place them locally before running the scripts.

Public release assets that can be redistributed are provided under `release_assets/`:

- `agiqa3k_split_seed42.json`: fixed AGIQA-3K split used by the main rerun
- `logic_feature_names.json`: fixed order of the 12-dimensional logic interface
- `logic_cache_schema.md`: field descriptions for the Stage A cache and compressed logic vector

Additional derived artifacts that depend on redistribution restrictions of the underlying benchmarks should be archived through Zenodo or provided by the corresponding author according to the manuscript statements.

## Expected Directory Layout

One workable local layout is:

```text
Prompt-Conditioned-Residual-Calibration/
├── release_assets/
├── runs/
├── outputs/
├── data/
│   ├── AGIQA-3K/
│   │   ├── agiqa.csv
│   │   └── images/
│   └── AIGCIQA2023/
│       ├── mytraindict_llm_2023.json
│       ├── mytestdict_llm_2023.json
│       ├── AIGIQA2023+.csv
│       └── images/
└── ...
```

Adapt the paths below to your local filesystem.

## Mainline Reproduction on AGIQA-3K

### 1. Run Stage A auditing

```bash
python verify_images_direct_alignment_features.py \
  --input_csv data/AGIQA-3K/agiqa.csv \
  --image_root data/AGIQA-3K/images \
  --output_jsonl outputs/agiqa/alignment_features_direct.jsonl \
  --output_csv outputs/agiqa/alignment_features_direct_raw.csv \
  --model YOUR_VLM_MODEL \
  --base_url YOUR_API_BASE \
  --api_key YOUR_API_KEY \
  --max_workers 2
```

This step exports requirement-level audit records for each prompt-image pair.

### 2. Compress Stage A outputs into the 12-dimensional logic interface

```bash
python vectorize_direct_alignment_features.py \
  --input_jsonl outputs/agiqa/alignment_features_direct.jsonl \
  --output_jsonl outputs/agiqa/alignment_logic_12d.jsonl \
  --output_csv outputs/agiqa/alignment_logic_12d.csv
```

### 3. Run the main controlled ablations

```bash
python run_model_ablations.py \
  --data_csv_path data/AGIQA-3K/agiqa.csv \
  --image_base_dir data/AGIQA-3K/images \
  --clip_model_name openai/clip-vit-large-patch14 \
  --funnel_cache_jsonl outputs/agiqa/alignment_logic_12d.jsonl \
  --output_root runs/agiqa_mainline \
  --split_seed 42 \
  --train_seeds 11,22,33
```

Important reported defaults:

- split seed: `42`
- train seeds: `11,22,33`
- epochs: `15`
- batch size: `16`
- learning rate: `2e-4`

### 4. Run a single fixed-split experiment directly

```bash
python train.py \
  --data_csv_path data/AGIQA-3K/agiqa.csv \
  --image_base_dir data/AGIQA-3K/images \
  --clip_model_name openai/clip-vit-large-patch14 \
  --device cuda \
  --epochs 15 \
  --batch_size 16 \
  --lr 2e-4 \
  --w_q 0.3 \
  --w_c 0.7 \
  --seed 11 \
  --split_file release_assets/agiqa3k_split_seed42.json \
  --output_dir runs/agiqa_single \
  --run_name full_model_seed11 \
  --funnel_cache_jsonl outputs/agiqa/alignment_logic_12d.jsonl
```

## Reproduction on the AIGCIQA2023 Extension Split

### 1. Build the local extension CSV and split JSON

```bash
python prepare_aigciqa2023_extension.py \
  --train_json data/AIGCIQA2023/mytraindict_llm_2023.json \
  --test_json data/AIGCIQA2023/mytestdict_llm_2023.json \
  --index_csv data/AIGCIQA2023/AIGIQA2023+.csv \
  --image_base_dir data/AIGCIQA2023/images \
  --output_csv outputs/aigciqa2023/aigciqa2023_extension.csv \
  --output_split_json outputs/aigciqa2023/aigciqa2023_extension_split.json
```

### 2. Run Stage A on the extension split

```bash
python verify_images_direct_alignment_features.py \
  --input_csv outputs/aigciqa2023/aigciqa2023_extension.csv \
  --image_root data/AIGCIQA2023/images \
  --output_jsonl outputs/aigciqa2023/alignment_features_direct.jsonl \
  --output_csv outputs/aigciqa2023/alignment_features_direct_raw.csv \
  --model YOUR_VLM_MODEL \
  --base_url YOUR_API_BASE \
  --api_key YOUR_API_KEY
```

### 3. Compress the extension cache

```bash
python vectorize_direct_alignment_features.py \
  --input_jsonl outputs/aigciqa2023/alignment_features_direct.jsonl \
  --output_jsonl outputs/aigciqa2023/alignment_logic_12d.jsonl \
  --output_csv outputs/aigciqa2023/alignment_logic_12d.csv
```

### 4. Run a fixed-split training or evaluation pass on the extension

```bash
python train.py \
  --data_csv_path outputs/aigciqa2023/aigciqa2023_extension.csv \
  --image_base_dir data/AIGCIQA2023/images \
  --clip_model_name openai/clip-vit-large-patch14 \
  --device cuda \
  --epochs 15 \
  --batch_size 16 \
  --lr 2e-4 \
  --w_q 0.3 \
  --w_c 0.7 \
  --seed 11 \
  --split_file outputs/aigciqa2023/aigciqa2023_extension_split.json \
  --output_dir runs/aigciqa2023_single \
  --run_name full_model_seed11 \
  --funnel_cache_jsonl outputs/aigciqa2023/alignment_logic_12d.jsonl
```

## External Baselines Under the Matched Local Protocol

The repository does not vendor third-party baseline code. Prepare the corresponding repositories separately, then supply their locations to the baseline runner.

```bash
python run_external_fair_baselines.py \
  --data_csv_path data/AGIQA-3K/agiqa.csv \
  --image_base_dir data/AGIQA-3K/images \
  --output_root runs/external_fair \
  --variants ipce,clip_agiqa,ma_agiqa \
  --external_repo_root _repo_inspect \
  --split_seed 42 \
  --train_seeds 11,22,33
```

## Direct Local VLM Judge

```bash
python run_local_vlm_judge.py \
  --data_csv_path outputs/aigciqa2023/aigciqa2023_extension.csv \
  --image_base_dir data/AIGCIQA2023/images \
  --output_dir runs/aigciqa2023_vlm_judge \
  --model YOUR_VLM_MODEL \
  --base_url YOUR_API_BASE \
  --api_key YOUR_API_KEY \
  --split_file outputs/aigciqa2023/aigciqa2023_extension_split.json \
  --split_role test
```

## Figure Generation

Generate the AIGCIQA2023 ablation bar chart:

```bash
python make_fig3_ablation_bars.py \
  --output_pdf Fig3_ablation_bars.pdf \
  --output_png Fig3_ablation_bars.png
```

Generate the AGIQA-3K residual scatter plot from an exported validation-prediction CSV:

```bash
python make_fig4_residual_scatter.py \
  --csv_path /path/to/val_preds.csv \
  --output_pdf Fig4_residual_scatter.pdf \
  --output_png Fig4_residual_scatter.png
```

## Expected Outputs

For a complete reproduction, you should retain at least the following:

- Stage A raw audit exports
- 12-dimensional logic-vector exports
- split JSON files
- run configuration JSON files
- validation and test prediction CSV files
- figure-generation inputs and outputs
- aggregate metric tables used in the manuscript

These files are the recommended candidates for Zenodo archiving together with the repository snapshot.

## Citation and Manuscript Linkage

This repository directly accompanies the manuscript below and should be cited together with the DOI-bearing software archive:

```bibtex
@article{qian2026nsrc,
  author  = {Qian, Siyuan},
  title   = {Neuro-Symbolic Residual Calibration for Fine-Grained Text-to-Image Alignment Assessment},
  journal = {The Visual Computer},
  year    = {2026},
  note    = {Manuscript submitted. Replace with the final bibliographic record after acceptance.}
}
```

Zenodo archive placeholder:

```bibtex
@software{qian2026pcrc,
  author = {Qian, Siyuan},
  title  = {Prompt-Conditioned Residual Calibration: Code and Reproduction Assets},
  year   = {2026},
  doi    = {<CODE_DOI>}
}
```
