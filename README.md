# Prompt-Conditioned Residual Calibration with Compact Logic Features for Text-to-Image Alignment Assessment

这个仓库整理自论文主线实现，面向 `ONSRC` 的公开发布版本。它只保留当前论文主线真正需要的脚本、最小依赖的 `src`、论文源文件，以及与可复现性直接相关的 release assets。

仓库地址：

`https://github.com/qiyan111/Prompt-Conditioned-Residual-Calibration`

## 仓库内容

- `train.py`: 主线训练与单次固定划分重跑入口。
- `run_model_ablations.py`: 论文主消融的控制变量重跑脚本。
- `verify_images_direct_alignment_features.py`: Stage A 结构化图文核验，输出 requirement-level JSON。
- `vectorize_direct_alignment_features.py`: 将 Stage A 输出压缩为固定 12 维 logic vector。
- `run_external_fair_baselines.py`: 外部公平重跑比较入口，需要另行准备外部 baseline 仓库。
- `external_baseline_adapter.py`: 外部 baseline 适配层。
- `prepare_aigciqa2023_extension.py`: AIGCIQA2023 扩展数据准备脚本。
- `run_local_vlm_judge.py`: 直接 VLM judge 对照脚本。
- `make_fig3_ablation_bars.py`, `make_fig4_residual_scatter.py`: 论文图生成脚本。
- `src/funnel.py`: 12 维 logic feature 的名称、别名与 cache 合并工具。
- `release_assets/`: 公开发布的 split definition、logic schema 与说明。
- `paper/`: 期刊稿 LaTeX 源文件与编译所需图片。

## 说明

这个 release 聚焦论文主线。部分脚本仍保留同一实现家族中的附加开关，便于做公平重跑和对照实验；论文正文只使用 README 中说明的主线配置，不依赖未写入论文的方法分支。

## 环境

建议使用 Python 3.10+。

```bash
pip install -r requirements.txt
```

核心依赖包括 `torch`, `transformers`, `peft`, `openai`, `pandas`, `scipy`, `scikit-learn`, `Pillow`, `tqdm`, `matplotlib`。

## 数据准备

- 仓库不分发 `AGIQA-3K` 和 `AIGCIQA2023` 原始数据。
- 请根据各数据集的官方分发方式自行获取图像与标注。
- 训练与评测时通过 `--data_csv_path` 和 `--image_base_dir` 指向本地数据。

公开的 release assets 位于 `release_assets/`：

- `agiqa3k_split_seed42.json`: 主稿 AGIQA-3K 固定划分文件。
- `logic_feature_names.json`: 12 维 logic interface 的固定顺序。
- `logic_cache_schema.md`: Stage A cache 与 12 维向量的字段说明。

## 主线复现

### 1. 运行 Stage A 核验并导出 12 维 logic vector

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

### 2. 运行论文主消融

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

如果只想做单次固定划分训练，也可以直接调用 `train.py`，并将 `--split_file` 指向 `release_assets/agiqa3k_split_seed42.json`。

### 3. AIGCIQA2023 扩展集

```bash
python prepare_aigciqa2023_extension.py \
  --train_json /path/to/mytraindict_llm_2023.json \
  --test_json /path/to/mytestdict_llm_2023.json \
  --image_base_dir /path/to/AIGCIQA2023 \
  --output_csv outputs/aigciqa2023_extension.csv \
  --output_split_json outputs/aigciqa2023_extension_split.json
```

## 外部 baseline

`run_external_fair_baselines.py` 用于在同一 split 协议下重跑外部 baseline。该脚本不内置第三方仓库代码；请先自行准备外部仓库，并通过 `--external_repo_root` 或各自的 `--*_repo_dir` 参数指定路径。

## 论文源文件

期刊稿源文件位于 `paper/`。如需重新编译：

```bash
cd paper
pdflatex paper_mainline_visual_computer.tex
bibtex paper_mainline_visual_computer
pdflatex paper_mainline_visual_computer.tex
pdflatex paper_mainline_visual_computer.tex
```
