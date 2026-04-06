#!/usr/bin/env python3


import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from transformers import CLIPModel, CLIPProcessor, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr
from src.funnel import FUNNEL_LOGIC_FEATURE_NAMES, merge_funnel_cache

# Optional PEFT (LoRA)
_PEFT_IMPORT_ERROR = ""
try:
    from peft import LoraConfig, TaskType, get_peft_model

    _PEFT_AVAILABLE = True
except Exception as exc:
    _PEFT_AVAILABLE = False
    _PEFT_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


class TrainingConfig:
    """Hyper-parameters with sensible defaults. Can be overridden via CLI."""

    def __init__(self):
        self.data_csv_path = ""  # path to the training CSV
        self.image_base_dir = ""  # root directory that stores the images
        self.clip_model_name = "openai/clip-vit-large-patch14"

        # training
        self.epochs = 20
        self.batch_size = 16
        self.lr = 2e-4
        self.weight_decay = 1e-4
        self.w_q = 0.3
        self.w_c = 0.7
        self.image_size = 224
        self.use_train_aug = False
        self.crop_scale_min = 0.8
        self.hflip_p = 0.5
        self.test_size = 0.2
        self.val_size_within_train = 0.1
        self.disable_validation_split = False
        self.train_loss_stop_threshold = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 42
        self.split_seed = 42
        self.split_file = ""
        self.output_dir = ""
        self.run_name = ""
        self.group_column = "auto"
        self.selection_metric = "weighted_qc_srocc"
        self.freeze_clip = False  # set True for linear probing baseline
        self.pure_linear_probe = False
        self.use_two_branch = True
        # explanation-based distillation (optional)
        self.use_explanations = False  # enable rationale loss if CSV has 'explanation'
        self.w_exp = 0.1  # weight for rationale alignment loss
        self.explanation_column = "explanation"  # CSV column name for explanation text

        # consistency refinement module (two-stage refinement)
        self.use_refinement = False  # enable refinement module
        self.refinement_layers = 4  # number of transformer layers in refinement (2-6 recommended)
        self.refinement_heads = 8  # number of attention heads (4-8 recommended)
        self.refinement_dim = 256  # hidden dimension for refinement module
        self.strict_residual = False  # use strict residual learning (supervise residual directly)

        # ===== residual learning =====
        self.use_residual_learning = True  # keep CLIP as the prior and predict residual corrections
        self.residual_scale_q = 0.2  # scaling factor for the quality residual
        self.residual_scale_c = 0.2  # scaling factor for the alignment residual
        self.partial_freeze = False  # finetune only later CLIP layers when enabled
        self.freeze_layers = 8  # freeze the first N ViT layers (ViT-L has 24 layers)
        # ===== RACL (Rank-Augmented Consistency Learning) =====
        self.use_rank_loss = False  # enable pairwise ranking loss
        self.rank_alpha = 10.0  # slope for ranking logits
        self.rank_pairs = 64  # sampled pair count per batch
        self.rank_lambda = 0.5  # ranking loss weight relative to MSE

        # ===== heteroscedastic regression (use std columns) =====
        self.use_hetero_weight = True
        self.std_floor = 1e-3
        self.hetero_weight_clip = 15.0

        # ===== group DRO over generator groups =====
        # The paper's default "full model" keeps GroupDRO off and enables it
        # only as a dedicated robustness ablation.
        self.use_group_dro = False
        self.group_dro_temp = 0.5
        self.group_dro_lambda = 0.0

        # ===== PCRC (Prompt-Conditioned Residual Calibration) =====
        self.use_pcrc = True
        self.use_film = True
        self.use_logic_concat = False
        self.pcrc_num_anchors = 16
        self.pcrc_hidden = 256
        self.pcrc_learnable_anchors = False
        self.pcrc_dynamic_anchors = False
        self.pcrc_anchor_texts = ""

        # ===== PromptMHA (token-level prompt attention) =====
        self.use_prompt_mha = True
        self.prompt_mha_heads = 8
        self.prompt_mha_dropout = 0.1

        # ===== Optional: G2R-MoE =====
        self.use_moe = False
        self.moe_num_experts = 4
        self.moe_gate_hidden = 256
        self.moe_tau = 1.0
        self.moe_entropy_lambda = 0.01

        # ===== PEFT: LoRA =====
        self.use_lora = True
        self.lora_r = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.05
        self.lora_target_modules = "q_proj,k_proj,v_proj,out_proj,visual_projection,text_projection"

        # ===== group-DRO mode =====
        self.group_dro_mode = "softmax_batch"  # ["softmax_batch", "expgrad"]
        self.group_dro_eta = 0.1

        # ===== Eval + explainability =====
        self.save_val_preds_csv = "val_preds.csv"
        self.save_token_importance_jsonl = "val_token_importance.jsonl"
        self.token_topk = 5
        self.bootstrap_iters = 2000
        self.bootstrap_seed = 42
        self.save_val_metrics_json = "val_metrics_ci.json"
        self.save_train_log_json = "train_log.json"
        self.save_config_json = "config.json"
        self.save_checkpoint_name = "checkpoint.pt"
        self.save_ig_topk = False
        self.ig_steps = 16
        self.ig_max_batches = 2
        self.funnel_cache_jsonl = ""
        self.text_source = "raw_prompt"
        self.use_focus_local_branch = True
        self.focus_local_scale = 1.0
        self.focus_local_text_source = "funnel_selected_prompt"
        self.focus_local_fallback_to_global = True
        self.group_metric_min_size = 3
        self.refit_on_trainval = True


FUNNEL_LOGIC_DIM = len(FUNNEL_LOGIC_FEATURE_NAMES)


def _name_prefix_group(name: Any) -> str:
    text = str(name).strip()
    if not text:
        return "unknown"
    return text.split("_")[0] or "unknown"


def _normalize_group_name(value: Any, fallback_name: Any) -> str:
    if value is None:
        return _name_prefix_group(fallback_name)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return _name_prefix_group(fallback_name)
    return text


def resolve_group_assignments(df: pd.DataFrame, group_column: str = "auto") -> Tuple[List[str], str]:
    priority = ["generator", "gen_model", "model"]
    if group_column != "auto":
        if group_column in {"name_prefix", "filename_prefix"}:
            return [_name_prefix_group(name) for name in df["name"].tolist()], "name_prefix"
        if group_column not in df.columns:
            raise ValueError(f"group_column='{group_column}' not found in dataframe.")
        groups = [
            _normalize_group_name(value, fallback_name=name)
            for value, name in zip(df[group_column].tolist(), df["name"].tolist())
        ]
        return groups, group_column

    for column in priority:
        if column in df.columns:
            valid = df[column].dropna().astype(str).str.strip()
            if (valid != "").any():
                groups = [
                    _normalize_group_name(value, fallback_name=name)
                    for value, name in zip(df[column].tolist(), df["name"].tolist())
                ]
                return groups, column
    return [_name_prefix_group(name) for name in df["name"].tolist()], "name_prefix"


def _safe_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


TEXT_SOURCE_CHOICES = (
    "raw_prompt",
    "funnel_text_prompt",
    "funnel_selected_prompt",
    "funnel_subject_text",
    "funnel_selected_phrase",
)


def resolve_text_source(row: pd.Series, text_source: str) -> str:
    raw_prompt = _safe_text_value(row.get("prompt"))
    candidates = {
        "raw_prompt": raw_prompt,
        "funnel_text_prompt": _safe_text_value(row.get("funnel_text_prompt")) or _safe_text_value(row.get("text_prompt")),
        "funnel_selected_prompt": _safe_text_value(row.get("funnel_selected_prompt")) or _safe_text_value(row.get("selected_prompt")),
        "funnel_subject_text": _safe_text_value(row.get("funnel_subject_text")) or _safe_text_value(row.get("main_visual_subject")),
        "funnel_selected_phrase": _safe_text_value(row.get("funnel_selected_phrase")) or _safe_text_value(row.get("selected_phrase")),
    }
    if text_source not in candidates:
        raise ValueError(f"Unsupported text_source='{text_source}'. Expected one of: {TEXT_SOURCE_CHOICES}")
    text = candidates[text_source]
    if text:
        return text
    if text_source == "raw_prompt":
        return ""
    raise ValueError(
        f"text_source='{text_source}' was requested, but the required funnel text field is missing for row "
        f"name='{_safe_text_value(row.get('name'))}'."
    )


def _clamp01_scalar(value: Any, default: float = 0.0) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        num = default
    if not math.isfinite(num):
        num = default
    return max(0.0, min(1.0, num))


def _prompt_token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.lower()))


def resolve_focus_text_source(
        row: pd.Series,
        text_source: str,
        fallback_to_global: bool = True) -> Tuple[str, str]:
    order = [text_source]
    for source in ("funnel_selected_prompt", "funnel_selected_phrase", "funnel_subject_text"):
        if source not in order:
            order.append(source)
    if fallback_to_global and "raw_prompt" not in order:
        order.append("raw_prompt")

    for source in order:
        try:
            text = resolve_text_source(row, source)
        except ValueError:
            continue
        if text:
            return text, source
    return "", text_source


def compute_focus_prompt_weight(full_prompt: str, focus_prompt: str, focus_text_source: str) -> float:
    full_prompt = _safe_text_value(full_prompt)
    focus_prompt = _safe_text_value(focus_prompt)
    if not focus_prompt:
        return 0.0
    if focus_text_source == "raw_prompt":
        return 1.0
    full_count = max(_prompt_token_count(full_prompt), 1)
    focus_count = max(_prompt_token_count(focus_prompt), 1)
    return max(0.0, min(1.0, focus_count / full_count))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _json_ready(value: Any) -> Any:
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    return str(value)


def training_config_to_dict(cfg: TrainingConfig) -> Dict[str, Any]:
    return {k: _json_ready(v) for k, v in vars(cfg).items() if not k.startswith("_")}


def ensure_run_dir(cfg: TrainingConfig) -> str:
    base_dir = cfg.output_dir.strip() if isinstance(cfg.output_dir, str) else ""
    run_name = cfg.run_name.strip() if isinstance(cfg.run_name, str) else ""
    if base_dir and run_name:
        run_dir = os.path.join(base_dir, run_name)
    elif base_dir:
        run_dir = base_dir
    else:
        run_dir = os.getcwd()
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def resolve_output_path(run_dir: str, path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(run_dir, path_value)


def ensure_split_row_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    out["_split_row_id"] = np.arange(len(out), dtype=np.int64)
    return out


PROMPT_COMPLEXITY_BUCKETS = (
    "subject_only",
    "attribute_heavy",
    "relation_count_heavy",
    "scene_style_heavy",
    "multi_object",
)


def infer_prompt_complexity_bucket(prompt: Any) -> str:
    text = _safe_text_value(prompt).lower()
    if not text:
        return "subject_only"

    tokens = set(re.findall(r"[a-z0-9]+", text))
    relation_keywords = {
        "with", "holding", "wearing", "beside", "behind", "between", "under", "over",
        "near", "next", "inside", "outside", "around", "riding", "standing", "sitting",
    }
    count_keywords = {
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "single", "pair", "double", "multiple", "several", "many", "few",
    }
    scene_style_keywords = {
        "background", "landscape", "city", "street", "room", "forest", "ocean", "sky",
        "sunset", "night", "indoors", "outdoors", "cinematic", "photorealistic", "anime",
        "cartoon", "watercolor", "oil", "painting", "sketch", "render", "lighting",
        "style", "scene", "atmosphere",
    }
    attribute_keywords = {
        "red", "blue", "green", "yellow", "black", "white", "gold", "silver", "brown",
        "wooden", "metallic", "cute", "small", "large", "big", "tiny", "giant",
        "ancient", "modern", "futuristic", "warm", "cold", "bright", "dark",
        "happy", "sad", "round", "square", "striped", "glossy", "matte",
    }
    multi_object_markers = [
        ",", " and ", " alongside ", " together ", " surrounded by ", " group of ",
    ]

    has_relation = bool(tokens.intersection(relation_keywords))
    has_count = bool(tokens.intersection(count_keywords)) or bool(re.search(r"\b\d+\b", text))
    if has_relation or has_count:
        return "relation_count_heavy"

    multi_object_score = sum(text.count(marker) for marker in multi_object_markers)
    if multi_object_score > 0:
        return "multi_object"

    if tokens.intersection(scene_style_keywords):
        return "scene_style_heavy"

    if len(tokens.intersection(attribute_keywords)) >= 2:
        return "attribute_heavy"

    return "subject_only"


def build_alignment_quantile_bins(values: pd.Series) -> pd.Series:
    if len(values) == 0:
        return pd.Series(dtype=object)

    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    default_label = "mid"
    out = pd.Series([default_label] * len(values), index=values.index, dtype=object)
    if valid.empty or valid.nunique() <= 1:
        return out

    rank = valid.rank(method="first")
    if valid.nunique() >= 3:
        labels = ["low", "mid", "high"]
        bins = pd.qcut(rank, q=3, labels=labels)
    else:
        labels = ["low", "high"]
        bins = pd.qcut(rank, q=2, labels=labels)
    out.loc[bins.index] = bins.astype(str)
    return out


def annotate_split_context(df: pd.DataFrame, group_column: str = "auto") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    annotated = df.copy()
    group_names, group_source = resolve_group_assignments(annotated, group_column=group_column)
    annotated["_split_generator_group"] = group_names
    annotated["_split_alignment_bin"] = build_alignment_quantile_bins(annotated["mos_align"]) if "mos_align" in annotated.columns else "mid"
    annotated["_split_prompt_complexity"] = [
        infer_prompt_complexity_bucket(prompt)
        for prompt in annotated["prompt"].tolist()
    ] if "prompt" in annotated.columns else ["subject_only"] * len(annotated)
    annotated["_eval_group_name"] = (
        annotated["_split_generator_group"].astype(str)
        + "|"
        + annotated["_split_alignment_bin"].astype(str)
        + "|"
        + annotated["_split_prompt_complexity"].astype(str)
    )
    return annotated, {
        "group_source": group_source,
        "alignment_source": "mos_align" if "mos_align" in annotated.columns else "missing",
        "prompt_complexity_source": "heuristic_prompt_complexity",
        "eval_group_definition": "generator|alignment_quantile|prompt_complexity",
    }


def _combine_stratify_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    if not columns:
        return pd.Series(["all"] * len(df), index=df.index, dtype=object)
    parts = [df[column].astype(str) for column in columns]
    out = parts[0].copy()
    for part in parts[1:]:
        out = out + "|" + part
    return out


def build_stratify_candidates(df: pd.DataFrame) -> List[Tuple[str, pd.Series]]:
    candidates: List[Tuple[str, pd.Series]] = []
    if {"_split_generator_group", "_split_alignment_bin", "_split_prompt_complexity"}.issubset(df.columns):
        candidates.append(
            (
                "generator+alignment+prompt_complexity",
                _combine_stratify_columns(df, ["_split_generator_group", "_split_alignment_bin", "_split_prompt_complexity"]),
            )
        )
    if {"_split_generator_group", "_split_alignment_bin"}.issubset(df.columns):
        candidates.append(
            (
                "generator+alignment",
                _combine_stratify_columns(df, ["_split_generator_group", "_split_alignment_bin"]),
            )
        )
    if {"_split_generator_group", "_split_prompt_complexity"}.issubset(df.columns):
        candidates.append(
            (
                "generator+prompt_complexity",
                _combine_stratify_columns(df, ["_split_generator_group", "_split_prompt_complexity"]),
            )
        )
    if {"_split_alignment_bin", "_split_prompt_complexity"}.issubset(df.columns):
        candidates.append(
            (
                "alignment+prompt_complexity",
                _combine_stratify_columns(df, ["_split_alignment_bin", "_split_prompt_complexity"]),
            )
        )
    if "_split_generator_group" in df.columns:
        candidates.append(("generator", df["_split_generator_group"].astype(str)))
    if "_split_alignment_bin" in df.columns:
        candidates.append(("alignment", df["_split_alignment_bin"].astype(str)))
    if "_split_prompt_complexity" in df.columns:
        candidates.append(("prompt_complexity", df["_split_prompt_complexity"].astype(str)))
    return candidates


def _split_ids_with_candidates(
        df: pd.DataFrame,
        heldout_size: float,
        random_state: int) -> Tuple[List[int], List[int], str]:
    row_ids = df["_split_row_id"].astype(int).tolist()
    candidates = build_stratify_candidates(df)
    for strategy_name, labels in candidates:
        try:
            train_ids, heldout_ids = train_test_split(
                row_ids,
                test_size=heldout_size,
                random_state=random_state,
                stratify=labels.astype(str).tolist(),
            )
            return (
                sorted(int(x) for x in train_ids),
                sorted(int(x) for x in heldout_ids),
                strategy_name,
            )
        except ValueError:
            continue

    train_ids, heldout_ids = train_test_split(row_ids, test_size=heldout_size, random_state=random_state)
    return (
        sorted(int(x) for x in train_ids),
        sorted(int(x) for x in heldout_ids),
        "random",
    )


def _validate_split_id_sets(
        available_ids: Sequence[int],
        train_ids: Sequence[int],
        val_ids: Sequence[int],
        test_ids: Sequence[int]) -> None:
    available = set(int(x) for x in available_ids)
    train_set = set(int(x) for x in train_ids)
    val_set = set(int(x) for x in val_ids)
    test_set = set(int(x) for x in test_ids)
    if train_set | val_set | test_set != available:
        raise ValueError("split_file does not match the current dataframe rows.")
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("split_file has overlapping train/val/test ids.")


def load_or_create_nested_split(
        df: pd.DataFrame,
        test_size: float,
        val_size_within_train: float,
        split_seed: int,
        split_file: str = "",
        split_metadata: Optional[Dict[str, Any]] = None,
        disable_validation_split: bool = False) -> Tuple[List[int], List[int], List[int], Dict[str, Any]]:
    available_ids = [int(x) for x in df["_split_row_id"].tolist()]
    split_path = split_file.strip()
    if split_path and os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if "test_ids" in payload:
            train_ids = [int(x) for x in payload["train_ids"]]
            val_ids = [int(x) for x in payload["val_ids"]]
            test_ids = [int(x) for x in payload["test_ids"]]
            _validate_split_id_sets(available_ids, train_ids, val_ids, test_ids)
            return sorted(train_ids), sorted(val_ids), sorted(test_ids), payload

        legacy_train_ids = [int(x) for x in payload["train_ids"]]
        legacy_test_ids = [int(x) for x in payload["val_ids"]]
        if disable_validation_split:
            upgraded_payload = {
                "version": 3,
                "split_seed": int(split_seed),
                "test_size": float(test_size),
                "val_size_within_train": 0.0,
                "disable_validation_split": True,
                "num_rows": int(len(df)),
                "train_ids": sorted(int(x) for x in legacy_train_ids),
                "val_ids": [],
                "test_ids": sorted(int(x) for x in legacy_test_ids),
                "outer_split_strategy": "legacy_train_test_file",
                "inner_split_strategy": "disabled_no_validation",
            }
            if split_metadata:
                upgraded_payload.update({k: _json_ready(v) for k, v in split_metadata.items()})
            if split_path:
                with open(split_path, "w", encoding="utf-8") as f:
                    json.dump(upgraded_payload, f, ensure_ascii=False, indent=2)
            return upgraded_payload["train_ids"], upgraded_payload["val_ids"], upgraded_payload["test_ids"], upgraded_payload
        train_pool_df = df[df["_split_row_id"].isin(legacy_train_ids)].reset_index(drop=True)
        inner_split_seed = int(split_seed + 1009)
        train_ids, val_ids, inner_strategy = _split_ids_with_candidates(
            train_pool_df,
            heldout_size=val_size_within_train,
            random_state=inner_split_seed,
        )
        upgraded_payload = {
            "version": 2,
            "split_seed": int(split_seed),
            "inner_split_seed": inner_split_seed,
            "test_size": float(test_size),
            "val_size_within_train": float(val_size_within_train),
            "num_rows": int(len(df)),
            "train_ids": sorted(int(x) for x in train_ids),
            "val_ids": sorted(int(x) for x in val_ids),
            "test_ids": sorted(int(x) for x in legacy_test_ids),
            "outer_split_strategy": "legacy_train_test_file",
            "inner_split_strategy": inner_strategy,
        }
        if split_metadata:
            upgraded_payload.update({k: _json_ready(v) for k, v in split_metadata.items()})
        if split_path:
            with open(split_path, "w", encoding="utf-8") as f:
                json.dump(upgraded_payload, f, ensure_ascii=False, indent=2)
        return upgraded_payload["train_ids"], upgraded_payload["val_ids"], upgraded_payload["test_ids"], upgraded_payload

    train_pool_ids, test_ids, outer_strategy = _split_ids_with_candidates(
        df,
        heldout_size=test_size,
        random_state=split_seed,
    )
    if disable_validation_split:
        payload = {
            "version": 3,
            "split_seed": int(split_seed),
            "test_size": float(test_size),
            "val_size_within_train": 0.0,
            "disable_validation_split": True,
            "effective_train_fraction": float(len(train_pool_ids) / max(len(df), 1)),
            "effective_val_fraction": 0.0,
            "effective_test_fraction": float(len(test_ids) / max(len(df), 1)),
            "num_rows": int(len(df)),
            "train_ids": sorted(int(x) for x in train_pool_ids),
            "val_ids": [],
            "test_ids": sorted(int(x) for x in test_ids),
            "outer_split_strategy": outer_strategy,
            "inner_split_strategy": "disabled_no_validation",
        }
        if split_metadata:
            payload.update({k: _json_ready(v) for k, v in split_metadata.items()})
        if split_path:
            split_parent = os.path.dirname(split_path)
            if split_parent:
                os.makedirs(split_parent, exist_ok=True)
            with open(split_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        return payload["train_ids"], payload["val_ids"], payload["test_ids"], payload
    train_pool_df = df[df["_split_row_id"].isin(train_pool_ids)].reset_index(drop=True)
    inner_split_seed = int(split_seed + 1009)
    train_ids, val_ids, inner_strategy = _split_ids_with_candidates(
        train_pool_df,
        heldout_size=val_size_within_train,
        random_state=inner_split_seed,
    )
    payload = {
        "version": 2,
        "split_seed": int(split_seed),
        "inner_split_seed": inner_split_seed,
        "test_size": float(test_size),
        "val_size_within_train": float(val_size_within_train),
        "effective_train_fraction": float(len(train_ids) / max(len(df), 1)),
        "effective_val_fraction": float(len(val_ids) / max(len(df), 1)),
        "effective_test_fraction": float(len(test_ids) / max(len(df), 1)),
        "num_rows": int(len(df)),
        "train_ids": sorted(int(x) for x in train_ids),
        "val_ids": sorted(int(x) for x in val_ids),
        "test_ids": sorted(int(x) for x in test_ids),
        "outer_split_strategy": outer_strategy,
        "inner_split_strategy": inner_strategy,
    }
    if split_metadata:
        payload.update({k: _json_ready(v) for k, v in split_metadata.items()})
    if split_path:
        split_parent = os.path.dirname(split_path)
        if split_parent:
            os.makedirs(split_parent, exist_ok=True)
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload["train_ids"], payload["val_ids"], payload["test_ids"], payload


def split_dataframe_from_three_way_ids(
        df: pd.DataFrame,
        train_ids: Sequence[int],
        val_ids: Sequence[int],
        test_ids: Sequence[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_set = set(int(x) for x in train_ids)
    val_set = set(int(x) for x in val_ids)
    test_set = set(int(x) for x in test_ids)
    train_df = df[df["_split_row_id"].isin(train_set)].sort_values("_split_row_id").reset_index(drop=True)
    val_df = df[df["_split_row_id"].isin(val_set)].sort_values("_split_row_id").reset_index(drop=True)
    test_df = df[df["_split_row_id"].isin(test_set)].sort_values("_split_row_id").reset_index(drop=True)
    return train_df, val_df, test_df


def load_or_create_split(
        df: pd.DataFrame,
        test_size: float,
        split_seed: int,
        split_file: str = "") -> Tuple[List[int], List[int], Dict[str, Any]]:
    available_ids = set(int(x) for x in df["_split_row_id"].tolist())
    split_path = split_file.strip()
    if split_path and os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        train_ids = [int(x) for x in payload["train_ids"]]
        val_ids = [int(x) for x in payload["val_ids"]]
        if set(train_ids).union(val_ids) != available_ids:
            raise ValueError("split_file does not match the current dataframe rows.")
        if set(train_ids).intersection(val_ids):
            raise ValueError("split_file has overlapping train/val ids.")
        return sorted(train_ids), sorted(val_ids), payload

    row_ids = sorted(available_ids)
    train_ids, val_ids = train_test_split(row_ids, test_size=test_size, random_state=split_seed)
    payload = {
        "split_seed": int(split_seed),
        "test_size": float(test_size),
        "num_rows": int(len(df)),
        "train_ids": sorted(int(x) for x in train_ids),
        "val_ids": sorted(int(x) for x in val_ids),
    }
    if split_path:
        split_parent = os.path.dirname(split_path)
        if split_parent:
            os.makedirs(split_parent, exist_ok=True)
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload["train_ids"], payload["val_ids"], payload


def split_dataframe_from_ids(df: pd.DataFrame, train_ids: Sequence[int], val_ids: Sequence[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_set = set(int(x) for x in train_ids)
    val_set = set(int(x) for x in val_ids)
    train_df = df[df["_split_row_id"].isin(train_set)].sort_values("_split_row_id").reset_index(drop=True)
    val_df = df[df["_split_row_id"].isin(val_set)].sort_values("_split_row_id").reset_index(drop=True)
    return train_df, val_df


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return int(sum(p.numel() for p in params))


def get_default_pcrc_anchor_texts() -> List[str]:
    """Default quality/alignment semantic anchors for PCRC."""
    return [
        "high quality image",
        "low quality image",
        "sharp details",
        "blurry artifacts",
        "natural colors",
        "color distortion",
        "clean background",
        "visual noise",
        "realistic rendering",
        "unnatural rendering",
        "prompt is well matched",
        "prompt is poorly matched",
        "good composition",
        "bad composition",
        "coherent semantics",
        "semantic inconsistency",
        "high perceptual quality",
        "low perceptual quality",
        "high fidelity textures",
        "texture smearing",
        "balanced lighting",
        "overexposed highlights",
        "underexposed shadows",
        "strong structural coherence",
        "severe structural artifacts",
        "accurate object attributes",
        "wrong object attributes",
        "clear foreground subject",
        "cluttered foreground subject",
        "prompt semantics are preserved",
        "prompt semantics are violated",
        "harmonious scene layout",
    ]


def build_pcrc_anchor_texts(num_anchors: int, anchor_texts_csv: str = "") -> List[str]:
    """Build anchor list from user CSV or defaults, then clip to num_anchors."""
    if num_anchors <= 0:
        raise ValueError("pcrc_num_anchors must be > 0.")
    if anchor_texts_csv.strip():
        anchors = [x.strip() for x in anchor_texts_csv.split(",") if x.strip()]
        if len(anchors) == 0:
            raise ValueError("pcrc_anchor_texts is empty after parsing.")
    else:
        anchors = get_default_pcrc_anchor_texts()

    if num_anchors > len(anchors):
        raise ValueError(
            f"Requested {num_anchors} anchors, but only {len(anchors)} are available. "
            f"Pass --pcrc_anchor_texts with at least {num_anchors} comma-separated prompts."
        )
    return anchors[:num_anchors]


def _parse_target_modules(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _find_matching_target_modules(model: nn.Module, requested: Sequence[str]) -> List[str]:
    names = [n for n, _ in model.named_modules()]
    found = set()
    for n in names:
        last = n.split(".")[-1]
        for t in requested:
            if last == t or n.endswith(f".{t}"):
                found.add(t)
    return [t for t in requested if t in found]


def apply_lora_to_clip(
        clip_model: CLIPModel,
        r: int,
        alpha: int,
        dropout: float,
        target_modules: Sequence[str]) -> Tuple[nn.Module, List[str]]:
    if not _PEFT_AVAILABLE:
        raise RuntimeError(
            "use_lora=True but peft is unavailable. "
            "This usually means peft is missing or incompatible with the current transformers version. "
            f"Import error: {_PEFT_IMPORT_ERROR or 'unknown error'}"
        )

    for p in clip_model.parameters():
        p.requires_grad = False

    requested = list(target_modules)
    matched = _find_matching_target_modules(clip_model, requested)
    if not matched:
        fallback = ["q_proj", "k_proj", "v_proj", "out_proj", "visual_projection", "text_projection"]
        matched = _find_matching_target_modules(clip_model, fallback)
    if not matched:
        raise RuntimeError(
            "No LoRA target_modules matched CLIP modules. "
            "Please set --lora_target_modules to valid module names."
        )

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=matched,
    )
    wrapped = get_peft_model(clip_model, cfg)
    return wrapped, matched


class BaselineDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            img_dir: str,
            proc: CLIPProcessor,
            tfm,
            focus_tfm=None,
            group_column: str = "auto",
            text_source: str = "raw_prompt",
            focus_text_source: str = "funnel_selected_prompt",
            focus_local_fallback_to_global: bool = True,
            use_focus_local_branch: bool = True):
        self.df, self.img_dir, self.proc, self.tfm = df.reset_index(drop=True).copy(), img_dir, proc, tfm
        self.focus_tfm = focus_tfm if focus_tfm is not None else tfm
        self.text_source = text_source
        self.focus_text_source = focus_text_source
        self.focus_local_fallback_to_global = focus_local_fallback_to_global
        self.use_focus_local_branch = use_focus_local_branch
        # optional explanation/rationale text column
        self.exp_col = "explanation" if "explanation" in self.df.columns else None
        # optional std columns for heteroscedastic weighting
        self.has_q_std = "std_quality" in self.df.columns
        self.has_c_std = "std_align" in self.df.columns
        # group id for group-DRO
        self.group_names, self.group_source = resolve_group_assignments(self.df, group_column=group_column)
        self.df["_resolved_group_name"] = self.group_names
        self.df["_resolved_group_source"] = self.group_source
        uniq = sorted(set(self.group_names))
        self.group2id = {g: i for i, g in enumerate(uniq)}
        self.group_ids = [self.group2id[g] for g in self.group_names]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, r["name"])).convert("RGB")
        px = self.tfm(img)
        text_input = resolve_text_source(r, self.text_source)
        toks = self.proc(
            text=[text_input],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        focus_px = px
        focus_text_input = text_input
        focus_weight = 0.0
        focus_valid = 0.0
        if self.use_focus_local_branch:
            focus_path = _safe_text_value(r.get("funnel_focus_image_path")) or _safe_text_value(r.get("focus_image_path"))
            focus_candidates = []
            if focus_path:
                focus_candidates.append(focus_path)
                if not os.path.isabs(focus_path):
                    focus_candidates.append(os.path.join(self.img_dir, focus_path))
            focus_img = None
            for candidate in focus_candidates:
                if candidate and os.path.exists(candidate):
                    focus_img = Image.open(candidate).convert("RGB")
                    break
            if focus_img is not None:
                focus_px = self.focus_tfm(focus_img)
                focus_valid = _clamp01_scalar(r.get("focus_valid"), default=1.0)
            focus_text_input, focus_text_used = resolve_focus_text_source(
                r,
                self.focus_text_source,
                fallback_to_global=self.focus_local_fallback_to_global,
            )
            focus_weight = compute_focus_prompt_weight(r.get("prompt"), focus_text_input, focus_text_used)
        focus_toks = self.proc(
            text=[focus_text_input],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        # optional explanation tokens (for distillation)
        if self.exp_col is not None:
            exp_text = str(r[self.exp_col])
            exp_tok = self.proc(
                text=[exp_text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77  # CLIP's max sequence length
            )
            exp_ids = exp_tok.input_ids[0]
            exp_mask = exp_tok.attention_mask[0]
        else:
            exp_ids = None
            exp_mask = None
        q_std = float(r["std_quality"]) if self.has_q_std else 1.0
        c_std = float(r["std_align"]) if self.has_c_std else 1.0
        if not math.isfinite(q_std) or q_std <= 0:
            q_std = 1.0
        if not math.isfinite(c_std) or c_std <= 0:
            c_std = 1.0
        group_id = self.group_ids[idx]
        logic_vec = r.get("funnel_logic_vec")
        if isinstance(logic_vec, str):
            try:
                logic_vec = json.loads(logic_vec)
            except Exception:
                logic_vec = [0.0] * FUNNEL_LOGIC_DIM
        if not isinstance(logic_vec, list) or len(logic_vec) != FUNNEL_LOGIC_DIM:
            logic_vec = [0.0] * FUNNEL_LOGIC_DIM
        logic_tensor = torch.tensor(logic_vec, dtype=torch.float32)
        return (
            str(r["name"]),
            px,
            toks.input_ids[0],
            toks.attention_mask[0],
            torch.tensor(r["mos_quality"] / 5., dtype=torch.float32),
            torch.tensor(r["mos_align"] / 5., dtype=torch.float32),
            torch.tensor(q_std / 5., dtype=torch.float32),
            torch.tensor(c_std / 5., dtype=torch.float32),
            torch.tensor(group_id, dtype=torch.long),
            exp_ids,
            exp_mask,
            logic_tensor,
            focus_px,
            focus_toks.input_ids[0],
            focus_toks.attention_mask[0],
            torch.tensor(focus_weight, dtype=torch.float32),
            torch.tensor(focus_valid, dtype=torch.float32),
        )


def collate_fn(batch):
    names = [b[0] for b in batch]
    px = torch.stack([b[1] for b in batch])
    ids = nn.utils.rnn.pad_sequence([b[2] for b in batch], batch_first=True)
    mask = nn.utils.rnn.pad_sequence([b[3] for b in batch], batch_first=True)
    q = torch.stack([b[4] for b in batch]).unsqueeze(1)
    c = torch.stack([b[5] for b in batch]).unsqueeze(1)
    q_std = torch.stack([b[6] for b in batch]).unsqueeze(1)
    c_std = torch.stack([b[7] for b in batch]).unsqueeze(1)
    group_id = torch.stack([b[8] for b in batch])
    # explanation tokens are optional (may be None)
    has_exp = all(b[9] is not None for b in batch)
    if has_exp:
        exp_ids = nn.utils.rnn.pad_sequence([b[9] for b in batch], batch_first=True)
        exp_mask = nn.utils.rnn.pad_sequence([b[10] for b in batch], batch_first=True)
    else:
        exp_ids = None
        exp_mask = None
    logic_vec = torch.stack([b[11] for b in batch])
    focus_px = torch.stack([b[12] for b in batch])
    focus_ids = nn.utils.rnn.pad_sequence([b[13] for b in batch], batch_first=True)
    focus_mask = nn.utils.rnn.pad_sequence([b[14] for b in batch], batch_first=True)
    focus_weight = torch.stack([b[15] for b in batch]).unsqueeze(1)
    focus_valid = torch.stack([b[16] for b in batch]).unsqueeze(1)
    return (
        names,
        px,
        ids,
        mask,
        q,
        c,
        q_std,
        c_std,
        group_id,
        exp_ids,
        exp_mask,
        logic_vec,
        focus_px,
        focus_ids,
        focus_mask,
        focus_weight,
        focus_valid,
    )


class PromptAnchorBank(nn.Module):
    """Build prompt-anchor condition vector p(x) from image-anchor similarities."""

    def __init__(
            self,
            clip_model: CLIPModel,
            clip_processor: CLIPProcessor,
            anchor_texts: Sequence[str],
            learnable: bool = False,
            dynamic_recompute: bool = False):
        super().__init__()
        if len(anchor_texts) == 0:
            raise ValueError("anchor_texts must not be empty when PCRC is enabled.")
        self.learnable = learnable
        self.dynamic_recompute = dynamic_recompute
        self.anchor_texts = list(anchor_texts)

        toks = clip_processor(
            text=list(anchor_texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        self.register_buffer("anchor_input_ids", toks.input_ids, persistent=False)
        self.register_buffer("anchor_attention_mask", toks.attention_mask, persistent=False)

        with torch.no_grad():
            anchor_u = clip_model.get_text_features(
                input_ids=self.anchor_input_ids,
                attention_mask=self.anchor_attention_mask,
            )
            anchor_u = F.normalize(anchor_u, dim=-1)
        if learnable:
            self.anchor_u = nn.Parameter(anchor_u)
        else:
            self.register_buffer("anchor_u", anchor_u)

    @property
    def num_anchors(self) -> int:
        return len(self.anchor_texts)

    def _dynamic_anchor_u(self, clip_model: CLIPModel) -> torch.Tensor:
        with torch.no_grad():
            anchor_u = clip_model.get_text_features(
                input_ids=self.anchor_input_ids,
                attention_mask=self.anchor_attention_mask,
            )
        return F.normalize(anchor_u, dim=-1)

    def get_anchor_u(self, clip_model: Optional[CLIPModel] = None) -> torch.Tensor:
        if self.learnable:
            return F.normalize(self.anchor_u, dim=-1)
        if self.dynamic_recompute:
            if clip_model is None:
                raise ValueError("clip_model is required when dynamic_recompute=True.")
            return self._dynamic_anchor_u(clip_model)
        return self.anchor_u

    def sim_vector(self, v_img: torch.Tensor, clip_model: Optional[CLIPModel] = None) -> torch.Tensor:
        """Return [B, A] similarities in [-1, 1]."""
        anchor_u = self.get_anchor_u(clip_model=clip_model).to(v_img.dtype)
        return v_img @ anchor_u.t()


class PromptMHAEncoder(nn.Module):
    """Token-level multi-head attention to build prompt vector and token importance."""

    def __init__(self, dt: int, d_out: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.q_qa = nn.Parameter(torch.zeros(1, 1, dt))
        nn.init.normal_(self.q_qa, std=0.02)
        self.mha = nn.MultiheadAttention(
            embed_dim=dt,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.proj = nn.Sequential(
            nn.Linear(dt, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, token_hidden: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, _, dt = token_hidden.shape
        q = self.q_qa.expand(bsz, 1, dt)
        key_padding_mask = None
        if attn_mask is not None:
            key_padding_mask = (attn_mask == 0)
        out, attn = self.mha(
            q,
            token_hidden,
            token_hidden,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        alpha = attn.mean(dim=1).squeeze(1)  # [B, L]
        p_prompt = self.proj(out.squeeze(1))  # [B, d_out]
        return p_prompt, alpha


class PCRCResidualHead(nn.Module):
    """Predict (delta_q, delta_c) = r([v, u, p_sim, p_prompt]) modulated by logic_vec."""

    def __init__(self, d_clip: int, num_anchors: int, d_prompt: int, hidden: int = 256, use_film: bool = True):
        super().__init__()
        in_dim = 2 * d_clip + num_anchors + d_prompt
        self.use_film = use_film

        if self.use_film:
            self.film_gen = nn.Sequential(
                nn.Linear(FUNNEL_LOGIC_DIM, 64),
                nn.GELU(),
                nn.Linear(64, in_dim * 2)
            )
            nn.init.zeros_(self.film_gen[-1].weight)
            nn.init.zeros_(self.film_gen[-1].bias)
        else:
            self.film_gen = None

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        # Start exactly from baseline behavior: residual output = 0.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
            self,
            v_img: torch.Tensor,
            u_txt: torch.Tensor,
            p_sim: torch.Tensor,
            p_prompt: torch.Tensor,
            logic_vec: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if logic_vec is None:
            logic_vec = torch.zeros(v_img.size(0), FUNNEL_LOGIC_DIM, device=v_img.device, dtype=v_img.dtype)

        z = torch.cat([v_img, u_txt, p_sim, p_prompt], dim=-1)
        if self.use_film and self.film_gen is not None:
            film_params = self.film_gen(logic_vec)
            gamma, beta = film_params.chunk(2, dim=-1)
            z_mod = z * (gamma + 1.0) + beta
        else:
            z_mod = z

        dq_dc = self.mlp(z_mod)
        dq, dc = dq_dc[:, :1], dq_dc[:, 1:]
        return dq, dc


class G2RMoE(nn.Module):
    """Optional gated residual mixture-of-experts over PCRC heads."""

    def __init__(self, expert_ctor, gate_in_dim: int, num_experts: int = 4, gate_hidden: int = 256):
        super().__init__()
        self.experts = nn.ModuleList([expert_ctor() for _ in range(num_experts)])
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, num_experts),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, gate_feat: torch.Tensor, expert_inputs: Tuple[torch.Tensor, ...], tau: float = 1.0):
        logits = self.gate(gate_feat) / max(tau, 1e-6)
        w = torch.softmax(logits, dim=-1)  # [B, M]
        dq_list, dc_list = [], []
        for expert in self.experts:
            dq, dc = expert(*expert_inputs)
            dq_list.append(dq)
            dc_list.append(dc)
        dq_all = torch.stack(dq_list, dim=-1)  # [B, 1, M]
        dc_all = torch.stack(dc_list, dim=-1)  # [B, 1, M]
        dq = (dq_all * w.unsqueeze(1)).sum(dim=-1)
        dc = (dc_all * w.unsqueeze(1)).sum(dim=-1)
        return dq, dc, w


class ConsistencyRefinementModule(nn.Module):
    """Two-stage consistency refinement using lightweight Transformer.

    Stage 1: Coarse prediction from CLIP similarity
    Stage 2: Fine-grained refinement that learns residual corrections

    This module corrects CLIP's high/low bias and improves prediction smoothness.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2, num_heads: int = 4):
        """
        Args:
            input_dim: Dimension of CLIP embeddings (e.g., 512 or 768)
            hidden_dim: Hidden dimension for transformer
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
        """
        super().__init__()

        # Project [img_emb, txt_emb, coarse_score] to hidden_dim
        # Input: [B, input_dim*2 + 1]
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim * 2 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Lightweight Transformer for refinement
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Residual prediction head: outputs correction term
        # We use residual learning: refined = coarse + residual
        self.residual_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Bounded residual in [-1, 1]
        )

        # Learnable residual scale (starts small for stability)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor, coarse_score: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_emb: [B, dim] - normalized image embeddings
            txt_emb: [B, dim] - normalized text embeddings
            coarse_score: [B, 1] - coarse consistency score from stage 1

        Returns:
            refined_score: [B, 1] - refined consistency score (coarse + residual)
        """
        # Concatenate all inputs
        x = torch.cat([img_emb, txt_emb, coarse_score], dim=1)  # [B, dim*2+1]

        # Project to hidden dimension
        x = self.input_proj(x)  # [B, hidden_dim]

        # Add sequence dimension for transformer (treat as single token)
        x = x.unsqueeze(1)  # [B, 1, hidden_dim]

        # Apply transformer refinement
        x = self.transformer(x)  # [B, 1, hidden_dim]

        # Remove sequence dimension
        x = x.squeeze(1)  # [B, hidden_dim]

        # Predict residual correction (bounded)
        residual = self.residual_head(x)  # [B, 1], range [-1, 1]
        residual = self.residual_scale * residual  # Scale down for stability

        # Residual learning: refined = coarse + residual
        refined_score = coarse_score + residual

        # Clip to valid range [0, 1]
        refined_score = torch.clamp(refined_score, 0.0, 1.0)

        return refined_score


class BaselineCLIPScore(nn.Module):
    """Use CLIP global image/text embeddings for quality and consistency prediction."""

    def __init__(self, clip_model_name: str, freeze: bool = False,
                 pure_linear_probe: bool = False,
                 use_refinement: bool = False, refinement_cfg: dict = None,
                 use_two_branch: bool = True, use_residual_learning: bool = True,
                 residual_scale_q: float = 0.2, residual_scale_c: float = 0.2,
                 partial_freeze: bool = False, freeze_layers: int = 8,
                 use_pcrc: bool = True, pcrc_hidden: int = 256,
                 use_film: bool = True,
                 use_logic_concat: bool = False,
                 pcrc_anchor_texts: Optional[Sequence[str]] = None,
                 pcrc_learnable_anchors: bool = False,
                 pcrc_dynamic_anchors: bool = False,
                 clip_processor: Optional[CLIPProcessor] = None,
                 use_prompt_mha: bool = True,
                 prompt_mha_heads: int = 8,
                 prompt_mha_dropout: float = 0.1,
                 use_moe: bool = False,
                 moe_num_experts: int = 4,
                 moe_gate_hidden: int = 256,
                 moe_tau: float = 1.0,
                 use_focus_local_branch: bool = True,
                 focus_local_scale: float = 1.0,
                 use_lora: bool = True,
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.05,
                 lora_target_modules: Optional[Sequence[str]] = None):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.use_pure_linear_probe = pure_linear_probe
        if self.use_pure_linear_probe:
            freeze = True
            use_lora = False
            use_refinement = False
            use_residual_learning = False
            use_two_branch = False
            use_focus_local_branch = False

        self.use_lora = use_lora
        if self.use_lora:
            req = list(lora_target_modules) if lora_target_modules is not None else []
            self.clip, matched = apply_lora_to_clip(
                self.clip,
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=req,
            )
            print(
                f"  [LoRA] enabled: r={lora_r}, alpha={lora_alpha}, "
                f"dropout={lora_dropout}, targets={matched}"
            )
            if freeze or partial_freeze:
                print("  [LoRA] ignoring freeze_clip/partial_freeze because LoRA is enabled.")
        else:
            if freeze:
                for p in self.clip.parameters():
                    p.requires_grad = False
            elif partial_freeze:
                self._partial_freeze_clip(freeze_layers)

        dim = self.clip.config.projection_dim
        t_dim = self.clip.config.text_config.hidden_size

        if self.use_pure_linear_probe:
            self.use_residual_learning = False
            self.use_refinement = False
            self.use_two_branch = False
            self.use_focus_local_branch = False
            self.focus_local_scale = focus_local_scale
            self.focus_gate_mlp = None
            self.use_pcrc = False
            self.use_prompt_mha = False
            self.use_moe = False
            self.residual_scale_q = residual_scale_q
            self.residual_scale_c = residual_scale_c
            self.prompt_mha = None
            self.q_head = nn.Linear(dim, 1)
            self.c_linear_head = nn.Linear(dim * 2 + 1, 1)
            self.refinement_module = None
            self.rationale_head = nn.Sequential(
                nn.LayerNorm(dim * 2 + 1),
                nn.Linear(dim * 2 + 1, dim)
            )
            return

        self.use_residual_learning = use_residual_learning
        self.residual_scale_q = residual_scale_q
        self.residual_scale_c = residual_scale_c
        self.use_focus_local_branch = use_focus_local_branch
        self.focus_local_scale = focus_local_scale
        if self.use_focus_local_branch:
            focus_gate_init = _clamp01_scalar(focus_local_scale, default=0.5)
            focus_gate_init = min(max(focus_gate_init, 1e-3), 1.0 - 1e-3)
            focus_gate_bias = math.log(focus_gate_init / (1.0 - focus_gate_init))
            self.focus_gate_mlp = nn.Sequential(
                nn.Linear(10, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
            nn.init.zeros_(self.focus_gate_mlp[-1].weight)
            nn.init.constant_(self.focus_gate_mlp[-1].bias, focus_gate_bias)
        else:
            self.focus_gate_mlp = None
        self.use_pcrc = use_residual_learning and use_pcrc
        self.use_film = use_film
        self.use_logic_concat = use_logic_concat and use_residual_learning and not self.use_pcrc
        self.use_prompt_mha = use_prompt_mha
        self.use_moe = use_moe
        self.moe_tau = moe_tau
        if use_pcrc and not use_residual_learning:
            print("  [PCRC] Disabled because residual learning is off.")
        if self.use_moe and not self.use_pcrc:
            print("  [MoE] Disabled because PCRC or residual learning is off.")
            self.use_moe = False

        if self.use_prompt_mha:
            self.prompt_mha = PromptMHAEncoder(
                dt=t_dim,
                d_out=dim,
                num_heads=prompt_mha_heads,
                dropout=prompt_mha_dropout,
            )
        else:
            self.prompt_mha = None

        if use_residual_learning:
            self.q_base_head = nn.Linear(dim, 1)
        else:
            self.q_head = nn.Linear(dim, 1)

        self.use_refinement = use_refinement
        self.use_two_branch = use_two_branch

        if use_residual_learning:
            self.c_base_scale = nn.Parameter(torch.tensor(1.0))
            self.c_base_bias = nn.Parameter(torch.tensor(0.0))

            if self.use_pcrc:
                if clip_processor is None:
                    raise ValueError("clip_processor is required when use_pcrc=True.")
                anchor_texts = list(pcrc_anchor_texts) if pcrc_anchor_texts is not None else get_default_pcrc_anchor_texts()
                self.anchor_bank = PromptAnchorBank(
                    clip_model=self.clip,
                    clip_processor=clip_processor,
                    anchor_texts=anchor_texts,
                    learnable=pcrc_learnable_anchors,
                    dynamic_recompute=pcrc_dynamic_anchors,
                )
                self.pcrc_head = PCRCResidualHead(
                    d_clip=dim,
                    num_anchors=self.anchor_bank.num_anchors,
                    d_prompt=dim,
                    hidden=pcrc_hidden,
                    use_film=self.use_film,
                )
                if self.use_moe:
                    gate_in_dim = 2 * dim + self.anchor_bank.num_anchors + dim

                    def expert_ctor():
                        return PCRCResidualHead(
                            d_clip=dim,
                            num_anchors=self.anchor_bank.num_anchors,
                            d_prompt=dim,
                            hidden=pcrc_hidden,
                            use_film=self.use_film,
                        )

                    self.moe = G2RMoE(
                        expert_ctor=expert_ctor,
                        gate_in_dim=gate_in_dim,
                        num_experts=moe_num_experts,
                        gate_hidden=moe_gate_hidden,
                    )
                else:
                    self.moe = None
            else:
                self.moe = None
                if self.use_logic_concat:
                    logic_concat_dim = dim * 2 + 1 + FUNNEL_LOGIC_DIM
                    self.logic_concat_head = nn.Sequential(
                        nn.LayerNorm(logic_concat_dim),
                        nn.Linear(logic_concat_dim, 256),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(256, 64),
                        nn.GELU(),
                        nn.Linear(64, 2),
                        nn.Tanh(),
                    )
                    nn.init.zeros_(self.logic_concat_head[-2].weight)
                    nn.init.zeros_(self.logic_concat_head[-2].bias)
                    self.q_delta_head = None
                    self.c_film_gen = None
                    self.c_delta_head = None
                else:
                    self.logic_concat_head = None
                    self.q_delta_head = nn.Sequential(
                        nn.Linear(dim, 128),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(128, 1),
                        nn.Tanh()
                    )
                    if self.use_film:
                        self.c_film_gen = nn.Sequential(
                            nn.Linear(FUNNEL_LOGIC_DIM, 64),
                            nn.GELU(),
                            nn.Linear(64, (dim * 2 + 1) * 2)
                        )
                        nn.init.zeros_(self.c_film_gen[-1].weight)
                        nn.init.zeros_(self.c_film_gen[-1].bias)
                    else:
                        self.c_film_gen = None
                    self.c_delta_head = nn.Sequential(
                        nn.LayerNorm(dim * 2 + 1),
                        nn.Linear(dim * 2 + 1, 256),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(256, 64),
                        nn.GELU(),
                        nn.Linear(64, 1),
                        nn.Tanh()
                    )
        else:
            self.c_scale = nn.Parameter(torch.tensor(5.0))
            self.c_bias = nn.Parameter(torch.tensor(0.0))

            self.filip_scale = nn.Parameter(torch.tensor(5.0))
            self.filip_bias = nn.Parameter(torch.tensor(0.0))

            v_dim = self.clip.config.vision_config.hidden_size
            t_dim = self.clip.config.text_config.hidden_size
            proj_dim = self.clip.config.projection_dim
            self.seq_proj_img = nn.Linear(v_dim, proj_dim, bias=False) if v_dim != proj_dim else nn.Identity()
            self.seq_proj_txt = nn.Linear(t_dim, proj_dim, bias=False) if t_dim != proj_dim else nn.Identity()

            self.c_head = nn.Sequential(
                nn.LayerNorm(dim * 2 + 1),
                nn.Linear(dim * 2 + 1, 256),
                nn.GELU(),
                nn.Linear(256, 1)
            )
            self.moe = None

        self.rationale_head = nn.Sequential(
            nn.LayerNorm(dim * 2 + 1),
            nn.Linear(dim * 2 + 1, dim)
        )

        if use_refinement and not use_residual_learning:
            refinement_cfg = refinement_cfg or {}
            self.refinement_module = ConsistencyRefinementModule(
                input_dim=dim,
                hidden_dim=refinement_cfg.get('hidden_dim', 256),
                num_layers=refinement_cfg.get('num_layers', 2),
                num_heads=refinement_cfg.get('num_heads', 4)
            )
        else:
            self.refinement_module = None

    def _partial_freeze_clip(self, freeze_layers: int):
        """Freeze first N layers of CLIP vision/text encoders."""
        print(f"  [Partial Freeze] Freezing first {freeze_layers} layers of CLIP Vision and Text encoders")

        if hasattr(self.clip.vision_model, 'encoder'):
            for i, layer in enumerate(self.clip.vision_model.encoder.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        if hasattr(self.clip.text_model, 'encoder'):
            for i, layer in enumerate(self.clip.text_model.encoder.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        if hasattr(self.clip.vision_model, 'embeddings'):
            for param in self.clip.vision_model.embeddings.parameters():
                param.requires_grad = False
        if hasattr(self.clip.text_model, 'embeddings'):
            for param in self.clip.text_model.embeddings.parameters():
                param.requires_grad = False

    def forward(
            self,
            pixel_values,
            ids,
            mask,
            logic_vec=None,
            focus_pixel_values=None,
            focus_ids=None,
            focus_mask=None,
            focus_weight=None,
            focus_valid=None,
            return_extras: bool = False):
        out = self.clip(pixel_values=pixel_values, input_ids=ids, attention_mask=mask, return_dict=True)
        img_g = F.normalize(out.image_embeds, dim=-1)
        txt_g = F.normalize(out.text_embeds, dim=-1)
        sim = (img_g * txt_g).sum(-1, keepdim=True)
        focus_sim = None
        focus_score = None
        focus_gate = None
        focus_img_g = None
        focus_txt_g = None
        focus_gate_learned = None

        if (
                self.use_focus_local_branch
                and focus_pixel_values is not None
                and focus_ids is not None
                and focus_mask is not None):
            if focus_valid is None:
                focus_valid = torch.ones_like(sim)
            else:
                focus_valid = focus_valid.view(-1, 1)
            if focus_weight is None:
                focus_weight = torch.ones_like(sim)
            else:
                focus_weight = focus_weight.view(-1, 1)
            if torch.any(focus_valid > 0):
                focus_out = self.clip(
                    pixel_values=focus_pixel_values,
                    input_ids=focus_ids,
                    attention_mask=focus_mask,
                    return_dict=True,
                )
                focus_img_g = F.normalize(focus_out.image_embeds, dim=-1)
                focus_txt_g = F.normalize(focus_out.text_embeds, dim=-1)
                focus_sim = (focus_img_g * focus_txt_g).sum(-1, keepdim=True)
                focus_score = torch.clamp((focus_sim + 1.0) / 2.0, 0.0, 1.0)
                focus_prior = torch.clamp(focus_valid * focus_weight, 0.0, 1.0)
                if self.focus_gate_mlp is not None:
                    global_score = torch.clamp((sim + 1.0) / 2.0, 0.0, 1.0)
                    img_focus_sim = (img_g * focus_img_g).sum(-1, keepdim=True)
                    txt_focus_sim = (txt_g * focus_txt_g).sum(-1, keepdim=True)
                    gate_features = torch.cat(
                        [
                            sim,
                            focus_sim,
                            global_score,
                            focus_score,
                            focus_prior,
                            focus_weight,
                            focus_valid,
                            img_focus_sim,
                            txt_focus_sim,
                            torch.abs(focus_score - global_score),
                        ],
                        dim=1,
                    )
                    focus_gate_learned = torch.sigmoid(self.focus_gate_mlp(gate_features))
                    focus_gate = torch.clamp(focus_prior * focus_gate_learned, 0.0, 1.0)
                else:
                    focus_gate = focus_prior

        if self.use_pure_linear_probe:
            fused = torch.cat([img_g, txt_g, sim], dim=1)
            q = torch.sigmoid(self.q_head(img_g))
            c_coarse = torch.sigmoid(self.c_linear_head(fused))
            c = c_coarse
            if return_extras:
                extras = {
                    "alpha": None,
                    "p_prompt": None,
                    "moe_w": None,
                    "token_hidden": None,
                    "p_sim": None,
                    "sim": sim,
                    "focus_sim": focus_sim,
                    "focus_score": focus_score,
                    "focus_gate": focus_gate,
                    "focus_gate_learned": focus_gate_learned,
                }
                return q, c, img_g, txt_g, c_coarse, extras
            return q, c, img_g, txt_g, c_coarse

        token_hidden = out.text_model_output.last_hidden_state if out.text_model_output is not None else None
        p_prompt = torch.zeros_like(img_g)
        alpha = None
        if self.prompt_mha is not None and token_hidden is not None:
            p_prompt, alpha = self.prompt_mha(token_hidden, mask)

        if self.use_residual_learning:
            q_base = torch.sigmoid(self.q_base_head(img_g))
        else:
            q = torch.sigmoid(self.q_head(img_g))

        if self.use_residual_learning:
            c_base = (sim + 1.0) / 2.0
            c_base = self.c_base_scale * c_base + self.c_base_bias
            c_base = torch.clamp(c_base, 0.0, 1.0)
            if focus_score is not None and focus_gate is not None:
                # Blend global and focus-local consistency so grounding contributes a direct CLIP score.
                c_base = torch.clamp((1.0 - focus_gate) * c_base + focus_gate * focus_score, 0.0, 1.0)

            if self.use_pcrc:
                p_sim = self.anchor_bank.sim_vector(
                    img_g,
                    clip_model=self.clip if self.anchor_bank.dynamic_recompute else None,
                )
                if self.use_moe and self.moe is not None:
                    gate_feat = torch.cat([img_g, txt_g, p_sim, p_prompt], dim=-1)
                    dq, dc, moe_w = self.moe(
                        gate_feat=gate_feat,
                        expert_inputs=(img_g, txt_g, p_sim, p_prompt, logic_vec),
                        tau=self.moe_tau,
                    )
                else:
                    dq, dc = self.pcrc_head(img_g, txt_g, p_sim, p_prompt, logic_vec)
                    moe_w = None
                q_delta = dq * self.residual_scale_q
                c_delta = dc * self.residual_scale_c
            else:
                if logic_vec is None:
                    logic_vec = torch.zeros(img_g.size(0), FUNNEL_LOGIC_DIM, device=img_g.device, dtype=img_g.dtype)

                fused = torch.cat([img_g, txt_g, sim], dim=1)
                if self.use_logic_concat and self.logic_concat_head is not None:
                    fused_logic = torch.cat([fused, logic_vec], dim=1)
                    dq_dc = self.logic_concat_head(fused_logic)
                    q_delta = dq_dc[:, :1] * self.residual_scale_q
                    c_delta = dq_dc[:, 1:] * self.residual_scale_c
                else:
                    q_delta = self.q_delta_head(img_g) * self.residual_scale_q
                    if self.use_film and self.c_film_gen is not None:
                        film_params = self.c_film_gen(logic_vec)
                        gamma, beta = film_params.chunk(2, dim=-1)
                        fused_mod = fused * (gamma + 1.0) + beta
                    else:
                        fused_mod = fused
                    c_delta = self.c_delta_head(fused_mod) * self.residual_scale_c
                p_sim = None
                moe_w = None

            q = torch.clamp(q_base + q_delta, 0.0, 1.0)
            c = torch.clamp(c_base + c_delta, 0.0, 1.0)
            c_coarse = c_base

        else:
            cos_score = torch.sigmoid(self.c_scale * sim + self.c_bias)
            if focus_score is not None and focus_gate is not None:
                cos_score = torch.clamp((1.0 - focus_gate) * cos_score + focus_gate * focus_score, 0.0, 1.0)

            img_seq = out.vision_model_output.last_hidden_state
            txt_seq = out.text_model_output.last_hidden_state

            img_seq = self.seq_proj_img(img_seq)
            txt_seq = self.seq_proj_txt(txt_seq)

            img_seq = F.normalize(img_seq, dim=-1)
            txt_seq = F.normalize(txt_seq, dim=-1)
            t2i = torch.max(torch.einsum('bid,bjd->bij', txt_seq, img_seq), dim=2).values.mean(1, keepdim=True)
            i2t = torch.max(torch.einsum('bid,bjd->bij', img_seq, txt_seq), dim=2).values.mean(1, keepdim=True)
            filip_sim = 0.5 * (t2i + i2t)
            filip_score = torch.sigmoid(self.filip_scale * filip_sim + self.filip_bias)

            fused = torch.cat([img_g, txt_g, sim], dim=1)
            mlp_score = torch.sigmoid(self.c_head(fused))

            if self.use_two_branch:
                c_coarse = 0.5 * (cos_score + mlp_score)
            else:
                c_coarse = mlp_score

            if self.use_refinement and self.refinement_module is not None:
                c = self.refinement_module(img_g, txt_g, c_coarse)
            else:
                c = c_coarse
            p_sim = None
            moe_w = None

        if return_extras:
            extras = {
                "alpha": alpha,
                "p_prompt": p_prompt,
                "moe_w": moe_w,
                "token_hidden": token_hidden,
                "p_sim": p_sim,
                "sim": sim,
                "focus_sim": focus_sim,
                "focus_score": focus_score,
                "focus_gate": focus_gate,
                "focus_gate_learned": focus_gate_learned,
            }
            return q, c, img_g, txt_g, c_coarse, extras
        return q, c, img_g, txt_g, c_coarse

    def compute_rationale_alignment_loss(self,
                                         img_g: torch.Tensor,
                                         txt_g: torch.Tensor,
                                         exp_ids: torch.Tensor,
                                         exp_mask: torch.Tensor) -> torch.Tensor:
        """Align fused [img, txt, cos] projection to explanation text embedding."""
        if exp_ids is None or exp_mask is None:
            return torch.tensor(0.0, device=img_g.device)
        with torch.no_grad():
            exp_g = self.clip.get_text_features(input_ids=exp_ids, attention_mask=exp_mask)
            exp_g = F.normalize(exp_g, dim=-1)
        sim = (img_g * txt_g).sum(-1, keepdim=True)
        fused = torch.cat([img_g, txt_g, sim], dim=1)
        pred = self.rationale_head(fused)
        pred = F.normalize(pred, dim=-1)
        return F.mse_loss(pred, exp_g)


def hetero_weighted_mse(err2: torch.Tensor, std: torch.Tensor, std_floor: float, clip_max: float):
    var = std.pow(2) + (std_floor ** 2)
    w = (1.0 / var).clamp(max=clip_max)
    w = w / (w.mean().detach() + 1e-8)
    return (w * err2).mean(), (w * err2)


class GroupDROState:
    def __init__(self, num_groups: int, eta: float, device: torch.device):
        self.eta = eta
        self.q = torch.ones(num_groups, device=device) / max(num_groups, 1)

    def update(self, group_losses: torch.Tensor, group_ids: torch.Tensor):
        with torch.no_grad():
            self.q[group_ids] = self.q[group_ids] * torch.exp(self.eta * group_losses)
            self.q = self.q / (self.q.sum() + 1e-12)

    def get_weights(self, group_ids: torch.Tensor):
        return self.q[group_ids]


def _safe_spearman(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    if y.size < 2 or yhat.size < 2:
        return 0.0
    r = spearmanr(y, yhat).correlation
    return 0.0 if (r is None or not np.isfinite(r)) else float(r)


def _safe_pearson(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    if y.size < 2 or yhat.size < 2:
        return 0.0
    r = pearsonr(y, yhat)[0]
    return 0.0 if (r is None or not np.isfinite(r)) else float(r)


def _safe_rmse(y, yhat):
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((yhat - y) ** 2)))


def _safe_mae(y, yhat):
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    if y.size == 0:
        return 0.0
    return float(np.mean(np.abs(yhat - y)))


def _metric_value(y, yhat, kind: str) -> float:
    if kind == "spearman":
        return _safe_spearman(y, yhat)
    if kind == "pearson":
        return _safe_pearson(y, yhat)
    if kind == "rmse":
        return _safe_rmse(y, yhat)
    if kind == "mae":
        return _safe_mae(y, yhat)
    raise ValueError(f"Unsupported metric kind: {kind}")


def bootstrap_ci_corr(y, yhat, kind="spearman", n_boot=2000, ci=0.95, seed=42):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    n = len(y)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    point = _safe_spearman(y, yhat) if kind == "spearman" else _safe_pearson(y, yhat)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        yh = yhat[idx]
        stats.append(_safe_spearman(yy, yh) if kind == "spearman" else _safe_pearson(yy, yh))
    stats = np.asarray(stats)
    lo = float(np.quantile(stats, (1 - ci) / 2))
    hi = float(np.quantile(stats, 1 - (1 - ci) / 2))
    return point, lo, hi


def paired_bootstrap_delta(
        y,
        yhat_ref,
        yhat_cmp,
        kind: str,
        n_boot: int = 2000,
        ci: float = 0.95,
        seed: int = 42) -> Dict[str, Any]:
    y = np.asarray(y)
    yhat_ref = np.asarray(yhat_ref)
    yhat_cmp = np.asarray(yhat_cmp)
    n = len(y)
    if n == 0:
        return {
            "reference": 0.0,
            "candidate": 0.0,
            "delta": 0.0,
            "ci95": [0.0, 0.0],
            "metric": kind,
        }
    ref_score = _metric_value(y, yhat_ref, kind)
    cmp_score = _metric_value(y, yhat_cmp, kind)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        delta = _metric_value(y[idx], yhat_cmp[idx], kind) - _metric_value(y[idx], yhat_ref[idx], kind)
        deltas.append(delta)
    deltas = np.asarray(deltas, dtype=np.float64)
    lo = float(np.quantile(deltas, (1 - ci) / 2))
    hi = float(np.quantile(deltas, 1 - (1 - ci) / 2))
    return {
        "reference": ref_score,
        "candidate": cmp_score,
        "delta": cmp_score - ref_score,
        "ci95": [lo, hi],
        "metric": kind,
    }


def compute_selection_score(
        metric_payload: Dict[str, Any],
        selection_metric: str,
        w_q: float = 0.3,
        w_c: float = 0.7) -> float:
    quality = metric_payload["quality"]["srocc"]["value"]
    consistency = metric_payload["consistency"]["srocc"]["value"]
    if selection_metric == "avg_srocc":
        return float(0.5 * (quality + consistency))
    if selection_metric == "weighted_qc_srocc":
        total = float(w_q + w_c)
        if total <= 0:
            return float(0.5 * (quality + consistency))
        return float((w_q / total) * quality + (w_c / total) * consistency)
    if selection_metric == "consistency_srocc":
        return float(consistency)
    if selection_metric == "quality_srocc":
        return float(quality)
    raise ValueError(f"Unsupported selection_metric='{selection_metric}'.")


def _token_topk(tokenizer, ids_row, mask_row, scores_row, topk):
    valid = mask_row.astype(bool)
    valid_ids = ids_row[valid]
    valid_scores = scores_row[valid]
    if valid_ids.shape[0] == 0:
        return [], []
    tokens = tokenizer.convert_ids_to_tokens(valid_ids.tolist())
    idx = np.argsort(valid_scores)[::-1][:topk]
    return [tokens[i] for i in idx], [float(valid_scores[i]) for i in idx]


def _compute_prompt_path_scores(model, img_g, txt_g, p_sim, token_hidden, mask, logic_vec=None):
    p_prompt, _ = model.prompt_mha(token_hidden, mask)
    if model.use_moe and model.moe is not None:
        gate_feat = torch.cat([img_g, txt_g, p_sim, p_prompt], dim=-1)
        dq, dc, _ = model.moe(
            gate_feat=gate_feat,
            expert_inputs=(img_g, txt_g, p_sim, p_prompt, logic_vec),
            tau=model.moe_tau,
        )
    else:
        dq, dc = model.pcrc_head(img_g, txt_g, p_sim, p_prompt, logic_vec)
    q_base = torch.sigmoid(model.q_base_head(img_g))
    sim = (img_g * txt_g).sum(-1, keepdim=True)
    c_base = (sim + 1.0) / 2.0
    c_base = model.c_base_scale * c_base + model.c_base_bias
    c_base = torch.clamp(c_base, 0.0, 1.0)
    q = torch.clamp(q_base + dq * model.residual_scale_q, 0.0, 1.0)
    c = torch.clamp(c_base + dc * model.residual_scale_c, 0.0, 1.0)
    return 0.5 * (q + c)


def _compute_prompt_path_ig_scores(model, img_g, txt_g, p_sim, token_hidden, mask, steps, logic_vec=None):
    baseline = torch.zeros_like(token_hidden)
    delta = token_hidden - baseline
    total_grad = torch.zeros_like(token_hidden)
    for i in range(1, max(steps, 1) + 1):
        a = float(i) / float(max(steps, 1))
        x = (baseline + a * delta).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        score = _compute_prompt_path_scores(model, img_g, txt_g, p_sim, x, mask, logic_vec=logic_vec).sum()
        grad = torch.autograd.grad(score, x, retain_graph=False, create_graph=False)[0]
        total_grad += grad.detach()
    avg_grad = total_grad / float(max(steps, 1))
    attr = delta * avg_grad
    return attr.abs().sum(dim=-1)


def _metric_block(y, yhat, bootstrap_iters: int, bootstrap_seed: int) -> Dict[str, Any]:
    s, s_lo, s_hi = bootstrap_ci_corr(y, yhat, kind="spearman", n_boot=bootstrap_iters, seed=bootstrap_seed)
    p, p_lo, p_hi = bootstrap_ci_corr(y, yhat, kind="pearson", n_boot=bootstrap_iters, seed=bootstrap_seed)
    return {
        "srocc": {"value": s, "ci95": [s_lo, s_hi]},
        "plcc": {"value": p, "ci95": [p_lo, p_hi]},
        "rmse": {"value": _safe_rmse(y, yhat)},
        "mae": {"value": _safe_mae(y, yhat)},
    }


def _compute_group_metrics(pred_df: pd.DataFrame, min_group_size: int = 3) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    group_column = "_eval_group_name" if "_eval_group_name" in pred_df.columns else "_resolved_group_name"
    if group_column not in pred_df.columns:
        return [], {
            "grouping_column": None,
            "min_group_size": int(min_group_size),
            "worst_group_c_srocc": 0.0,
            "worst_group_c_rmse": 0.0,
            "mean_group_c_srocc": 0.0,
        }

    rows: List[Dict[str, Any]] = []
    for group_name, gdf in pred_df.groupby(group_column, sort=True):
        rows.append(
            {
                "group_name": str(group_name),
                "count": int(len(gdf)),
                "eligible": bool(len(gdf) >= min_group_size),
                "consistency_srocc": _safe_spearman(gdf["target_c"].values, gdf["pred_c"].values),
                "consistency_plcc": _safe_pearson(gdf["target_c"].values, gdf["pred_c"].values),
                "consistency_rmse": _safe_rmse(gdf["target_c"].values, gdf["pred_c"].values),
            }
        )
    if not rows:
        return rows, {
            "grouping_column": group_column,
            "min_group_size": int(min_group_size),
            "worst_group_c_srocc": 0.0,
            "worst_group_c_rmse": 0.0,
            "mean_group_c_srocc": 0.0,
        }
    eligible_rows = [row for row in rows if row["eligible"]]
    summary_rows = eligible_rows if eligible_rows else rows
    summary = {
        "grouping_column": group_column,
        "min_group_size": int(min_group_size),
        "num_groups": int(len(rows)),
        "num_eligible_groups": int(len(eligible_rows)),
        "worst_group_c_srocc": float(min(row["consistency_srocc"] for row in summary_rows)),
        "worst_group_c_rmse": float(max(row["consistency_rmse"] for row in summary_rows)),
        "mean_group_c_srocc": float(np.mean([row["consistency_srocc"] for row in summary_rows])),
    }
    return rows, summary

def train_epoch(model, dl, opt, sched, cfg, dro_state=None):
    model.train()
    totals = [0, 0, 0, 0, 0, 0]
    for batch in dl:
        (
            _,
            px,
            ids,
            mask,
            q_t,
            c_t,
            q_std,
            c_std,
            group_id,
            exp_ids,
            exp_mask,
            logic_vec,
            focus_px,
            focus_ids,
            focus_mask,
            focus_weight,
            focus_valid,
        ) = batch
        px = px.to(cfg.device)
        ids = ids.to(cfg.device)
        mask = mask.to(cfg.device)
        q_t = q_t.to(cfg.device)
        c_t = c_t.to(cfg.device)
        q_std = q_std.to(cfg.device)
        c_std = c_std.to(cfg.device)
        group_id = group_id.to(cfg.device)
        logic_vec = logic_vec.to(cfg.device)
        focus_px = focus_px.to(cfg.device)
        focus_ids = focus_ids.to(cfg.device)
        focus_mask = focus_mask.to(cfg.device)
        focus_weight = focus_weight.to(cfg.device)
        focus_valid = focus_valid.to(cfg.device)
        if exp_ids is not None:
            exp_ids = exp_ids.to(cfg.device)
            exp_mask = exp_mask.to(cfg.device)

        opt.zero_grad()
        if cfg.use_moe:
            q_p, c_p, img_g, txt_g, c_coarse, extras = model(
                px,
                ids,
                mask,
                logic_vec=logic_vec,
                focus_pixel_values=focus_px,
                focus_ids=focus_ids,
                focus_mask=focus_mask,
                focus_weight=focus_weight,
                focus_valid=focus_valid,
                return_extras=True,
            )
        else:
            q_p, c_p, img_g, txt_g, c_coarse = model(
                px,
                ids,
                mask,
                logic_vec=logic_vec,
                focus_pixel_values=focus_px,
                focus_ids=focus_ids,
                focus_mask=focus_mask,
                focus_weight=focus_weight,
                focus_valid=focus_valid,
            )
            extras = None

        q_err2 = (q_p - q_t).pow(2)
        c_err2 = (c_p - c_t).pow(2)
        if cfg.use_hetero_weight:
            lq, q_err2 = hetero_weighted_mse(q_err2, q_std, cfg.std_floor, cfg.hetero_weight_clip)
            lc_base, c_err2 = hetero_weighted_mse(c_err2, c_std, cfg.std_floor, cfg.hetero_weight_clip)
        else:
            lq = q_err2.mean()
            lc_base = c_err2.mean()

        if cfg.use_refinement and model.use_refinement and cfg.strict_residual:
            target_residual = c_t - c_coarse.detach()
            predicted_residual = c_p - c_coarse.detach()
            lc = F.mse_loss(predicted_residual, target_residual)
        else:
            lc = lc_base

        lrq = torch.tensor(0.0, device=cfg.device)
        lrc = torch.tensor(0.0, device=cfg.device)
        if cfg.use_rank_loss:
            batch_size = q_p.size(0)
            num_pairs = min(cfg.rank_pairs, batch_size * (batch_size - 1) // 2)
            if batch_size > 1 and num_pairs > 0:
                idx1 = torch.randint(0, batch_size, (num_pairs,), device=cfg.device)
                idx2 = torch.randint(0, batch_size, (num_pairs,), device=cfg.device)
                mask_same = idx1 == idx2
                idx2[mask_same] = (idx2[mask_same] + 1) % batch_size
                tgt_diff_q = torch.sign(q_t[idx1] - q_t[idx2])
                pred_diff_q = q_p[idx1] - q_p[idx2]
                lrq = F.softplus(-cfg.rank_alpha * pred_diff_q * tgt_diff_q).mean()
                tgt_diff_c = torch.sign(c_t[idx1] - c_t[idx2])
                pred_diff_c = c_p[idx1] - c_p[idx2]
                lrc = F.softplus(-cfg.rank_alpha * pred_diff_c * tgt_diff_c).mean()

        if cfg.use_group_dro:
            sample_lq = q_err2.squeeze(1)
            sample_lc = c_err2.squeeze(1)
            sample_loss = cfg.w_q * sample_lq + cfg.w_c * sample_lc
            uniq_g = torch.unique(group_id)
            g_losses = torch.stack([sample_loss[group_id == g].mean() for g in uniq_g])
            if cfg.group_dro_mode == "expgrad" and dro_state is not None:
                dro_state.update(g_losses.detach(), uniq_g)
                g_w = dro_state.get_weights(uniq_g).detach()
            else:
                g_center = g_losses - g_losses.mean()
                g_w = torch.softmax(g_center / max(cfg.group_dro_temp, 1e-6), dim=0)
            robust_loss = torch.sum(g_w * g_losses)
            mean_loss = sample_loss.mean()
            loss = (1.0 - cfg.group_dro_lambda) * mean_loss + cfg.group_dro_lambda * robust_loss
        else:
            loss = cfg.w_q * lq + cfg.w_c * lc
        if cfg.use_rank_loss:
            loss = loss + cfg.rank_lambda * (lrq + lrc)

        if cfg.use_moe and extras is not None and extras.get("moe_w") is not None:
            w = extras["moe_w"]
            ent = -(w * torch.log(w + 1e-12)).sum(dim=-1).mean()
            loss = loss - cfg.moe_entropy_lambda * ent

        le = torch.tensor(0.0, device=cfg.device)
        if cfg.use_explanations and exp_ids is not None:
            le = model.compute_rationale_alignment_loss(img_g, txt_g, exp_ids, exp_mask)
            loss = loss + cfg.w_exp * le

        loss.backward()
        opt.step()
        sched.step()
        totals[0] += loss.item()
        totals[1] += lq.item()
        totals[2] += lc.item()
        totals[3] += le.item() if torch.is_tensor(le) else 0.0
        totals[4] += lrq.item() if torch.is_tensor(lrq) else 0.0
        totals[5] += lrc.item() if torch.is_tensor(lrc) else 0.0
    n = len(dl)
    return [t / n for t in totals]


def evaluate(
        model,
        dl,
        cfg,
        processor: CLIPProcessor,
        print_examples: bool = False,
        num_examples: int = 5,
        persist: bool = True,
        extra_metadata: Optional[Dict[str, Any]] = None,
        return_payload: bool = False):
    model.eval()
    names_all, preds_q, tgts_q, preds_c, tgts_c, preds_c_coarse = [], [], [], [], [], []
    token_records = []
    ig_batches_done = 0
    need_token_export = bool(cfg.save_token_importance_jsonl)
    need_extras = cfg.use_moe or need_token_export or cfg.save_ig_topk

    for batch in dl:
        (
            names,
            px,
            ids,
            mask,
            q_t,
            c_t,
            _,
            _,
            _,
            _,
            _,
            logic_vec,
            focus_px,
            focus_ids,
            focus_mask,
            focus_weight,
            focus_valid,
        ) = batch
        px = px.to(cfg.device)
        ids = ids.to(cfg.device)
        mask = mask.to(cfg.device)
        logic_vec = logic_vec.to(cfg.device)
        focus_px = focus_px.to(cfg.device)
        focus_ids = focus_ids.to(cfg.device)
        focus_mask = focus_mask.to(cfg.device)
        focus_weight = focus_weight.to(cfg.device)
        focus_valid = focus_valid.to(cfg.device)
        with torch.no_grad():
            if need_extras:
                q_p, c_p, img_g, txt_g, c_coarse, extras = model(
                    px,
                    ids,
                    mask,
                    logic_vec=logic_vec,
                    focus_pixel_values=focus_px,
                    focus_ids=focus_ids,
                    focus_mask=focus_mask,
                    focus_weight=focus_weight,
                    focus_valid=focus_valid,
                    return_extras=True,
                )
            else:
                q_p, c_p, img_g, txt_g, c_coarse = model(
                    px,
                    ids,
                    mask,
                    logic_vec=logic_vec,
                    focus_pixel_values=focus_px,
                    focus_ids=focus_ids,
                    focus_mask=focus_mask,
                    focus_weight=focus_weight,
                    focus_valid=focus_valid,
                )
                extras = None

        names_all.extend(names)
        preds_q.extend(q_p.detach().cpu().numpy().flatten())
        tgts_q.extend(q_t.detach().cpu().numpy().flatten())
        preds_c.extend(c_p.detach().cpu().numpy().flatten())
        tgts_c.extend(c_t.detach().cpu().numpy().flatten())
        preds_c_coarse.extend(c_coarse.detach().cpu().numpy().flatten())

        if need_token_export:
            ids_np = ids.detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy()
            alpha = extras.get("alpha") if extras is not None else None
            alpha_np = alpha.detach().cpu().numpy() if alpha is not None else None

            ig_np = None
            if (
                    cfg.save_ig_topk
                    and ig_batches_done < cfg.ig_max_batches
                    and extras is not None
                    and extras.get("token_hidden") is not None
                    and extras.get("p_sim") is not None
                    and model.use_residual_learning
                    and model.use_pcrc
                    and model.prompt_mha is not None):
                with torch.enable_grad():
                    ig_scores = _compute_prompt_path_ig_scores(
                        model=model,
                        img_g=img_g.detach(),
                        txt_g=txt_g.detach(),
                        p_sim=extras["p_sim"].detach(),
                        token_hidden=extras["token_hidden"].detach(),
                        mask=mask,
                        steps=cfg.ig_steps,
                        logic_vec=logic_vec.detach(),
                    )
                ig_np = ig_scores.detach().cpu().numpy()
                ig_batches_done += 1

            for i, name in enumerate(names):
                rec: Dict[str, Any] = {"name": name}
                if alpha_np is not None:
                    toks, scores = _token_topk(processor.tokenizer, ids_np[i], mask_np[i], alpha_np[i], cfg.token_topk)
                    rec["topk_tokens"] = toks
                    rec["topk_scores"] = scores
                    rec["method"] = "attention_mean_heads"
                if ig_np is not None:
                    ig_toks, ig_scores = _token_topk(processor.tokenizer, ids_np[i], mask_np[i], ig_np[i], cfg.token_topk)
                    rec["ig_topk_tokens"] = ig_toks
                    rec["ig_topk_scores"] = ig_scores
                    rec["ig_method"] = "prompt_path_ig"
                token_records.append(rec)

    pred_df = pd.DataFrame(
        {
            "name": names_all,
            "target_q": tgts_q,
            "pred_q": preds_q,
            "target_c": tgts_c,
            "pred_c": preds_c,
            "pred_c_coarse": preds_c_coarse,
        }
    )
    if hasattr(dl, "dataset") and hasattr(dl.dataset, "df"):
        meta_df = dl.dataset.df.reset_index(drop=True)
        if len(meta_df) == len(pred_df):
            meta_cols = [
                c for c in [
                    "_split_row_id",
                    "_resolved_group_name",
                    "_resolved_group_source",
                    "_split_generator_group",
                    "_split_alignment_bin",
                    "_split_prompt_complexity",
                    "_eval_group_name",
                    "generator",
                    "gen_model",
                    "model",
                ]
                if c in meta_df.columns
            ]
            if meta_cols:
                pred_df = pd.concat([pred_df, meta_df[meta_cols]], axis=1)

    quality_metrics = _metric_block(tgts_q, preds_q, cfg.bootstrap_iters, cfg.bootstrap_seed)
    consistency_metrics = _metric_block(tgts_c, preds_c, cfg.bootstrap_iters, cfg.bootstrap_seed)
    group_metrics, group_summary = _compute_group_metrics(pred_df, min_group_size=cfg.group_metric_min_size)

    paired_delta: Dict[str, Any] = {}
    pred_delta = np.asarray(preds_c, dtype=np.float64) - np.asarray(preds_c_coarse, dtype=np.float64)
    if cfg.use_refinement or np.any(np.abs(pred_delta) > 1e-12):
        paired_delta["consistency_refined_vs_coarse"] = {
            "srocc": paired_bootstrap_delta(
                tgts_c, preds_c_coarse, preds_c, kind="spearman",
                n_boot=cfg.bootstrap_iters, seed=cfg.bootstrap_seed
            ),
            "plcc": paired_bootstrap_delta(
                tgts_c, preds_c_coarse, preds_c, kind="pearson",
                n_boot=cfg.bootstrap_iters, seed=cfg.bootstrap_seed
            ),
            "rmse": paired_bootstrap_delta(
                tgts_c, preds_c_coarse, preds_c, kind="rmse",
                n_boot=cfg.bootstrap_iters, seed=cfg.bootstrap_seed
            ),
        }

    metrics = {
        "quality": quality_metrics,
        "consistency": {
            **consistency_metrics,
            "group_summary": group_summary,
        },
        "group_metrics": group_metrics,
        "paired_delta": paired_delta,
        "bootstrap": {"iters": cfg.bootstrap_iters, "seed": cfg.bootstrap_seed},
    }
    metrics["selection_score"] = compute_selection_score(
        metrics,
        cfg.selection_metric,
        w_q=cfg.w_q,
        w_c=cfg.w_c,
    )
    if extra_metadata:
        metrics.update({k: _json_ready(v) for k, v in extra_metadata.items()})

    if persist:
        pred_df.to_csv(cfg.save_val_preds_csv, index=False, encoding="utf-8")
        with open(cfg.save_val_metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        if need_token_export:
            with open(cfg.save_token_importance_jsonl, "w", encoding="utf-8") as f:
                for rec in token_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if print_examples and cfg.use_refinement:
        print(f"\n  {'=' * 70}")
        print("  Example Predictions (Coarse vs Refined vs Ground Truth):")
        print(f"  {'=' * 70}")
        print(f"  {'ID':<5} {'Coarse':>10} {'Refined':>10} {'GT':>10} {'Delta(Ref-Coa)':>14} {'Error':>10}")
        print(f"  {'-' * 70}")
        indices = list(range(min(num_examples, len(tgts_c))))
        for i in indices:
            coarse_val = preds_c_coarse[i]
            refined_val = preds_c[i]
            gt_val = tgts_c[i]
            delta = refined_val - coarse_val
            error = abs(refined_val - gt_val)
            print(
                f"  {i + 1:<5} {coarse_val:>10.4f} {refined_val:>10.4f} {gt_val:>10.4f} "
                f"{delta:>+12.4f} {error:>10.4f}"
            )
        deltas = [preds_c[i] - preds_c_coarse[i] for i in range(len(preds_c))]
        avg_delta = sum(deltas) / len(deltas)
        max_delta = max(deltas)
        min_delta = min(deltas)
        print(f"  {'-' * 70}")
        print(f"  Refinement Stats: Avg Delta={avg_delta:+.4f}, Min Delta={min_delta:+.4f}, Max Delta={max_delta:+.4f}")
        print(f"  {'=' * 70}\n")

    print(
        f"[Eval] Q(SROCC={quality_metrics['srocc']['value']:.4f},95%CI=[{quality_metrics['srocc']['ci95'][0]:.4f},"
        f"{quality_metrics['srocc']['ci95'][1]:.4f}], PLCC={quality_metrics['plcc']['value']:.4f}, "
        f"RMSE={quality_metrics['rmse']['value']:.4f}) | "
        f"C(SROCC={consistency_metrics['srocc']['value']:.4f},95%CI=[{consistency_metrics['srocc']['ci95'][0]:.4f},"
        f"{consistency_metrics['srocc']['ci95'][1]:.4f}], PLCC={consistency_metrics['plcc']['value']:.4f}, "
        f"RMSE={consistency_metrics['rmse']['value']:.4f})"
    )

    s_q = quality_metrics["srocc"]["value"]
    p_q = quality_metrics["plcc"]["value"]
    s_c = consistency_metrics["srocc"]["value"]
    p_c = consistency_metrics["plcc"]["value"]
    if return_payload:
        return s_q, p_q, s_c, p_c, metrics
    return s_q, p_q, s_c, p_c


def run_module_smoke_tests():
    bsz, seq_len, dt, d_clip, num_anchors = 2, 10, 32, 16, 6
    token_hidden = torch.randn(bsz, seq_len, dt)
    mask = torch.ones(bsz, seq_len, dtype=torch.long)
    prompt_mha = PromptMHAEncoder(dt=dt, d_out=d_clip, num_heads=4, dropout=0.1)
    p_prompt, alpha = prompt_mha(token_hidden, mask)
    assert p_prompt.shape == (bsz, d_clip), f"Unexpected p_prompt shape: {p_prompt.shape}"
    assert alpha.shape == (bsz, seq_len), f"Unexpected alpha shape: {alpha.shape}"

    head = PCRCResidualHead(d_clip=d_clip, num_anchors=num_anchors, d_prompt=d_clip, hidden=32)
    assert torch.allclose(head.mlp[-1].weight, torch.zeros_like(head.mlp[-1].weight))
    assert torch.allclose(head.mlp[-1].bias, torch.zeros_like(head.mlp[-1].bias))

    v_img = torch.randn(bsz, d_clip)
    u_txt = torch.randn(bsz, d_clip)
    p_sim = torch.randn(bsz, num_anchors)
    dq, dc = head(v_img, u_txt, p_sim, p_prompt)
    assert dq.shape == (bsz, 1), f"Unexpected dq shape: {dq.shape}"
    assert dc.shape == (bsz, 1), f"Unexpected dc shape: {dc.shape}"
    print("[SmokeTest] PromptMHAEncoder/PCRCResidualHead passed.")


def canonicalize_training_config(cfg: TrainingConfig) -> TrainingConfig:
    if cfg.pure_linear_probe:
        cfg.freeze_clip = True
        cfg.use_lora = False
        cfg.use_residual_learning = False
        cfg.use_refinement = False
        cfg.use_two_branch = False
        cfg.use_pcrc = False
        cfg.use_prompt_mha = False
        cfg.use_moe = False
        cfg.use_focus_local_branch = False
    if cfg.train_loss_stop_threshold is not None:
        try:
            threshold = float(cfg.train_loss_stop_threshold)
        except (TypeError, ValueError):
            threshold = None
        cfg.train_loss_stop_threshold = threshold if threshold is not None and threshold > 0 else None
    return cfg


def _build_shuffled_dataloader(dataset: Dataset, batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=generator,
    )


def _build_training_stack(
        cfg: TrainingConfig,
        proc: CLIPProcessor,
        train_dl: DataLoader,
        num_groups: int,
        num_epochs: int):
    refinement_cfg = {
        "hidden_dim": cfg.refinement_dim,
        "num_layers": cfg.refinement_layers,
        "num_heads": cfg.refinement_heads,
    } if cfg.use_refinement else None

    pcrc_anchor_texts = None
    if cfg.use_residual_learning and cfg.use_pcrc:
        pcrc_anchor_texts = build_pcrc_anchor_texts(
            num_anchors=cfg.pcrc_num_anchors,
            anchor_texts_csv=cfg.pcrc_anchor_texts,
        )
    lora_target_modules = _parse_target_modules(cfg.lora_target_modules)

    model = BaselineCLIPScore(
        cfg.clip_model_name,
        freeze=cfg.freeze_clip,
        pure_linear_probe=cfg.pure_linear_probe,
        use_refinement=cfg.use_refinement,
        refinement_cfg=refinement_cfg,
        use_two_branch=cfg.use_two_branch,
        use_residual_learning=cfg.use_residual_learning,
        residual_scale_q=cfg.residual_scale_q,
        residual_scale_c=cfg.residual_scale_c,
        partial_freeze=cfg.partial_freeze,
        freeze_layers=cfg.freeze_layers,
        use_pcrc=cfg.use_pcrc,
        pcrc_hidden=cfg.pcrc_hidden,
        use_film=cfg.use_film,
        use_logic_concat=cfg.use_logic_concat,
        pcrc_anchor_texts=pcrc_anchor_texts,
        pcrc_learnable_anchors=cfg.pcrc_learnable_anchors,
        pcrc_dynamic_anchors=cfg.pcrc_dynamic_anchors,
        clip_processor=proc,
        use_prompt_mha=cfg.use_prompt_mha,
        prompt_mha_heads=cfg.prompt_mha_heads,
        prompt_mha_dropout=cfg.prompt_mha_dropout,
        use_moe=cfg.use_moe,
        moe_num_experts=cfg.moe_num_experts,
        moe_gate_hidden=cfg.moe_gate_hidden,
        moe_tau=cfg.moe_tau,
        use_focus_local_branch=cfg.use_focus_local_branch,
        focus_local_scale=cfg.focus_local_scale,
        use_lora=cfg.use_lora,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_target_modules=lora_target_modules,
    ).to(cfg.device)
    trainable_params = count_parameters(model, trainable_only=True)
    total_params = count_parameters(model, trainable_only=False)

    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(len(train_dl) * max(int(num_epochs), 1), 1)
    warmup_steps = int(0.05 * total_steps)
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    dro_state = None
    if cfg.use_group_dro and cfg.group_dro_mode == "expgrad":
        dro_state = GroupDROState(num_groups=num_groups, eta=cfg.group_dro_eta, device=cfg.device)

    return model, opt, sched, dro_state, trainable_params, total_params


def run_training(cfg: TrainingConfig) -> Dict[str, Any]:
    cfg = canonicalize_training_config(cfg)
    if not (0.0 < float(cfg.test_size) < 1.0):
        raise ValueError(f"test_size must be in (0, 1), got {cfg.test_size}.")
    if (not cfg.disable_validation_split) and not (0.0 < float(cfg.val_size_within_train) < 1.0):
        raise ValueError(
            f"val_size_within_train must be in (0, 1), got {cfg.val_size_within_train}."
        )
    set_global_seed(cfg.seed)

    run_dir = ensure_run_dir(cfg)
    cfg.save_val_preds_csv = resolve_output_path(run_dir, cfg.save_val_preds_csv)
    cfg.save_val_metrics_json = resolve_output_path(run_dir, cfg.save_val_metrics_json)
    cfg.save_train_log_json = resolve_output_path(run_dir, cfg.save_train_log_json)
    cfg.save_config_json = resolve_output_path(run_dir, cfg.save_config_json)
    checkpoint_path = resolve_output_path(run_dir, cfg.save_checkpoint_name)
    selection_checkpoint_path = resolve_output_path(
        run_dir,
        f"selection_{os.path.basename(cfg.save_checkpoint_name)}",
    )
    if cfg.save_token_importance_jsonl:
        cfg.save_token_importance_jsonl = resolve_output_path(run_dir, cfg.save_token_importance_jsonl)
    if cfg.split_file:
        cfg.split_file = resolve_output_path(run_dir, cfg.split_file) if not os.path.isabs(cfg.split_file) else cfg.split_file

    with open(cfg.save_config_json, "w", encoding="utf-8") as f:
        json.dump(training_config_to_dict(cfg), f, ensure_ascii=False, indent=2)

    start_time = time.time()
    df = pd.read_csv(cfg.data_csv_path)
    if cfg.funnel_cache_jsonl:
        df = merge_funnel_cache(df, cfg.funnel_cache_jsonl)
    df = ensure_split_row_ids(df)
    df, split_annotation_meta = annotate_split_context(df, group_column=cfg.group_column)
    train_ids, val_ids, test_ids, split_payload = load_or_create_nested_split(
        df,
        test_size=cfg.test_size,
        val_size_within_train=cfg.val_size_within_train,
        split_seed=cfg.split_seed,
        split_file=cfg.split_file,
        split_metadata=split_annotation_meta,
        disable_validation_split=cfg.disable_validation_split,
    )
    train_df, val_df, test_df = split_dataframe_from_three_way_ids(df, train_ids, val_ids, test_ids)
    has_validation_split = (not cfg.disable_validation_split) and len(val_df) > 0

    proc = CLIPProcessor.from_pretrained(cfg.clip_model_name)
    train_tfms = []
    if cfg.use_train_aug:
        train_tfms.extend([
            transforms.RandomResizedCrop(cfg.image_size, scale=(cfg.crop_scale_min, 1.0)),
            transforms.RandomHorizontalFlip(p=cfg.hflip_p),
        ])
    else:
        train_tfms.append(transforms.Resize((cfg.image_size, cfg.image_size)))
    train_tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(proc.image_processor.image_mean, proc.image_processor.image_std),
    ])
    tf_train = transforms.Compose(train_tfms)
    tf_val = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(proc.image_processor.image_mean, proc.image_processor.image_std)
    ])

    if cfg.explanation_column != "explanation" and cfg.explanation_column in train_df.columns:
        train_df = train_df.rename(columns={cfg.explanation_column: "explanation"})
        val_df = val_df.rename(columns={cfg.explanation_column: "explanation"})
        test_df = test_df.rename(columns={cfg.explanation_column: "explanation"})

    train_ds = BaselineDataset(
        train_df,
        cfg.image_base_dir,
        proc,
        tf_train,
        focus_tfm=tf_val,
        group_column=cfg.group_column,
        text_source=cfg.text_source,
        focus_text_source=cfg.focus_local_text_source,
        focus_local_fallback_to_global=cfg.focus_local_fallback_to_global,
        use_focus_local_branch=cfg.use_focus_local_branch,
    )
    val_ds = BaselineDataset(
        val_df,
        cfg.image_base_dir,
        proc,
        tf_val,
        focus_tfm=tf_val,
        group_column=cfg.group_column,
        text_source=cfg.text_source,
        focus_text_source=cfg.focus_local_text_source,
        focus_local_fallback_to_global=cfg.focus_local_fallback_to_global,
        use_focus_local_branch=cfg.use_focus_local_branch,
    )
    test_ds = BaselineDataset(
        test_df,
        cfg.image_base_dir,
        proc,
        tf_val,
        focus_tfm=tf_val,
        group_column=cfg.group_column,
        text_source=cfg.text_source,
        focus_text_source=cfg.focus_local_text_source,
        focus_local_fallback_to_global=cfg.focus_local_fallback_to_global,
        use_focus_local_branch=cfg.use_focus_local_branch,
    )
    train_dl = _build_shuffled_dataloader(train_ds, cfg.batch_size, cfg.seed)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    model, opt, sched, dro_state, trainable_params, total_params = _build_training_stack(
        cfg,
        proc,
        train_dl,
        num_groups=len(train_ds.group2id),
        num_epochs=cfg.epochs,
    )

    print(f"\n{'=' * 70}")
    print("Training Configuration:")
    print(f"  - Run: {cfg.run_name or Path(run_dir).name}")
    print(f"  - Output Dir: {run_dir}")
    print(
        f"  - Split: seed={cfg.split_seed}, file={cfg.split_file or '<memory>'}, "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    print(
        f"    outer={split_payload.get('outer_split_strategy', 'unknown')}, "
        f"inner={split_payload.get('inner_split_strategy', 'unknown')}, "
        f"eval_groups={split_payload.get('eval_group_definition', 'generator')}"
    )
    print(f"  - Seed: {cfg.seed}")
    print(f"  - Epochs: {cfg.epochs}, Batch Size: {cfg.batch_size}, LR: {cfg.lr}")
    print(f"  - Loss Weights: w_q={cfg.w_q}, w_c={cfg.w_c}")
    print(f"  - Group DRO: {cfg.use_group_dro} (source={train_ds.group_source}, mode={cfg.group_dro_mode}, temp={cfg.group_dro_temp}, lambda={cfg.group_dro_lambda})")
    print(f"  - Pure Linear Probe: {cfg.pure_linear_probe}")
    print(f"  - LoRA: {cfg.use_lora}")
    print(f"  - Selection Metric: {cfg.selection_metric}")
    print(f"  - Train-Loss Early Stop Threshold: {cfg.train_loss_stop_threshold}")
    print(f"  - FiLM: {cfg.use_film}")
    print(f"  - Logic Concat: {cfg.use_logic_concat}")
    print(
        f"  - Focus Local Branch: {cfg.use_focus_local_branch} "
        f"(text={cfg.focus_local_text_source}, dynamic_gate_init={cfg.focus_local_scale}, "
        f"fallback_global={cfg.focus_local_fallback_to_global})"
    )
    print(f"  - Params: trainable={trainable_params}, total={total_params}")
    print(f"{'=' * 70}\n")

    best_score = -float("inf")
    best_epoch = -1
    last_train_loss = None
    best_train_loss = None
    stopped_on_train_loss_threshold = False
    history: List[Dict[str, Any]] = []
    selection_based_on = "validation" if has_validation_split else "last_epoch_train_loss"
    train_loss_stop_threshold = (
        float(cfg.train_loss_stop_threshold)
        if cfg.train_loss_stop_threshold is not None else None
    )

    for ep in range(cfg.epochs):
        train_loss, lq, lc, le, lrq, lrc = train_epoch(model, train_dl, opt, sched, cfg, dro_state=dro_state)
        last_train_loss = float(train_loss)
        best_train_loss = last_train_loss if best_train_loss is None else min(best_train_loss, last_train_loss)
        epoch_record = {
            "epoch": ep + 1,
            "train_loss": float(train_loss),
            "loss_q": float(lq),
            "loss_c": float(lc),
            "loss_exp": float(le),
            "loss_rank_q": float(lrq),
            "loss_rank_c": float(lrc),
        }
        history.append(epoch_record)
        if has_validation_split:
            print_examples = cfg.use_refinement and ((ep + 1) % 5 == 0 or (ep + 1) == cfg.epochs)
            s_q, p_q, s_c, p_c, eval_payload = evaluate(
                model,
                val_dl,
                cfg,
                proc,
                print_examples=print_examples,
                num_examples=5,
                persist=False,
                return_payload=True,
            )
            selection_score = float(eval_payload["selection_score"])
            epoch_record.update(
                {
                    "val_q_srocc": float(s_q),
                    "val_q_plcc": float(p_q),
                    "val_c_srocc": float(s_c),
                    "val_c_plcc": float(p_c),
                    "selection_score": selection_score,
                }
            )

            if cfg.use_rank_loss:
                if cfg.use_explanations:
                    print(
                        f"Ep{ep + 1} TrainLoss={train_loss:.4f}(Q{lq:.4f},C{lc:.4f},E{le:.4f},RankQ{lrq:.4f},RankC{lrc:.4f}) "
                        f"Val SROCC_Q={s_q:.4f},SROCC_C={s_c:.4f},Sel={selection_score:.4f}"
                    )
                else:
                    print(
                        f"Ep{ep + 1} TrainLoss={train_loss:.4f}(Q{lq:.4f},C{lc:.4f},RankQ{lrq:.4f},RankC{lrc:.4f}) "
                        f"Val SROCC_Q={s_q:.4f},SROCC_C={s_c:.4f},Sel={selection_score:.4f}"
                    )
            else:
                if cfg.use_explanations:
                    print(
                        f"Ep{ep + 1} TrainLoss={train_loss:.4f}(Q{lq:.4f},C{lc:.4f},E{le:.4f}) "
                        f"Val SROCC_Q={s_q:.4f},SROCC_C={s_c:.4f},Sel={selection_score:.4f}"
                    )
                else:
                    print(
                        f"Ep{ep + 1} TrainLoss={train_loss:.4f}(Q{lq:.4f},C{lc:.4f}) "
                        f"Val SROCC_Q={s_q:.4f},SROCC_C={s_c:.4f},Sel={selection_score:.4f}"
                    )

            if selection_score > best_score:
                best_score = selection_score
                best_epoch = ep + 1
                torch.save(model.state_dict(), selection_checkpoint_path)
                print(f"[Checkpoint] New best selection={best_score:.4f} -> {selection_checkpoint_path}")
        else:
            if cfg.use_rank_loss:
                if cfg.use_explanations:
                    print(
                        f"Ep{ep + 1} TrainLoss={train_loss:.4f}(Q{lq:.4f},C{lc:.4f},E{le:.4f},RankQ{lrq:.4f},RankC{lrc:.4f})"
                    )
                else:
                    print(
                        f"Ep{ep + 1} TrainLoss={train_loss:.4f}(Q{lq:.4f},C{lc:.4f},RankQ{lrq:.4f},RankC{lrc:.4f})"
                    )
            else:
                if cfg.use_explanations:
                    print(f"Ep{ep + 1} TrainLoss={train_loss:.4f}(Q{lq:.4f},C{lc:.4f},E{le:.4f})")
                else:
                    print(f"Ep{ep + 1} TrainLoss={train_loss:.4f}(Q{lq:.4f},C{lc:.4f})")

            if train_loss_stop_threshold is not None and float(train_loss) <= train_loss_stop_threshold:
                best_epoch = ep + 1
                stopped_on_train_loss_threshold = True
                selection_based_on = "train_loss_threshold"
                torch.save(model.state_dict(), checkpoint_path)
                print(
                    f"[Early Stop] TrainLoss={train_loss:.6f} <= {train_loss_stop_threshold:.6f} "
                    f"at epoch {best_epoch}; saved checkpoint -> {checkpoint_path}"
                )
                break

    if not has_validation_split and not stopped_on_train_loss_threshold:
        best_epoch = len(history)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[Checkpoint] Saved final epoch model (no validation split) -> {checkpoint_path}")

    refit_history: List[Dict[str, Any]] = []
    refit_payload: Dict[str, Any] = {
        "enabled": bool(has_validation_split and cfg.refit_on_trainval),
        "performed": False,
        "from_scratch": False,
        "epochs": 0,
        "combined_train_size": int(len(train_df)),
        "selection_checkpoint_path": selection_checkpoint_path if has_validation_split else "",
        "final_checkpoint_path": checkpoint_path,
        "status": "disabled_no_validation" if not has_validation_split else "disabled_by_config",
    }

    if has_validation_split:
        if not os.path.exists(selection_checkpoint_path):
            raise RuntimeError(
                f"Validation split is enabled, but no selection checkpoint was saved at {selection_checkpoint_path}."
            )
        state = torch.load(selection_checkpoint_path, map_location=cfg.device)
        model.load_state_dict(state)
        _, _, _, _, selected_val_payload = evaluate(
            model,
            val_dl,
            cfg,
            proc,
            print_examples=False,
            num_examples=5,
            persist=False,
            extra_metadata={
                "evaluation_split": "validation",
                "split_role_counts": {
                    "train": len(train_df),
                    "validation": len(val_df),
                    "test": len(test_df),
                },
            },
            return_payload=True,
        )
        selection_based_on = "validation"

        if cfg.refit_on_trainval:
            refit_epochs = max(int(best_epoch), 1)
            refit_df = pd.concat([train_df, val_df], ignore_index=True)
            refit_ds = BaselineDataset(
                refit_df,
                cfg.image_base_dir,
                proc,
                tf_train,
                focus_tfm=tf_val,
                group_column=cfg.group_column,
                text_source=cfg.text_source,
                focus_text_source=cfg.focus_local_text_source,
                focus_local_fallback_to_global=cfg.focus_local_fallback_to_global,
                use_focus_local_branch=cfg.use_focus_local_branch,
            )
            refit_dl = _build_shuffled_dataloader(refit_ds, cfg.batch_size, cfg.seed)

            print(
                f"[Refit] Re-training from scratch on train+validation for {refit_epochs} epochs "
                f"(train={len(train_df)}, val={len(val_df)}, combined={len(refit_df)})."
            )
            set_global_seed(cfg.seed)
            model, opt, sched, dro_state, trainable_params, total_params = _build_training_stack(
                cfg,
                proc,
                refit_dl,
                num_groups=len(refit_ds.group2id),
                num_epochs=refit_epochs,
            )
            for ep in range(refit_epochs):
                train_loss, lq, lc, le, lrq, lrc = train_epoch(
                    model,
                    refit_dl,
                    opt,
                    sched,
                    cfg,
                    dro_state=dro_state,
                )
                refit_record = {
                    "epoch": ep + 1,
                    "train_loss": float(train_loss),
                    "loss_q": float(lq),
                    "loss_c": float(lc),
                    "loss_exp": float(le),
                    "loss_rank_q": float(lrq),
                    "loss_rank_c": float(lrc),
                }
                refit_history.append(refit_record)
                if cfg.use_rank_loss:
                    if cfg.use_explanations:
                        print(
                            f"[Refit] Ep{ep + 1} TrainLoss={train_loss:.4f}"
                            f"(Q{lq:.4f},C{lc:.4f},E{le:.4f},RankQ{lrq:.4f},RankC{lrc:.4f})"
                        )
                    else:
                        print(
                            f"[Refit] Ep{ep + 1} TrainLoss={train_loss:.4f}"
                            f"(Q{lq:.4f},C{lc:.4f},RankQ{lrq:.4f},RankC{lrc:.4f})"
                        )
                else:
                    if cfg.use_explanations:
                        print(f"[Refit] Ep{ep + 1} TrainLoss={train_loss:.4f}(Q{lq:.4f},C{lc:.4f},E{le:.4f})")
                    else:
                        print(f"[Refit] Ep{ep + 1} TrainLoss={train_loss:.4f}(Q{lq:.4f},C{lc:.4f})")

            torch.save(model.state_dict(), checkpoint_path)
            print(f"[Refit] Saved final refit checkpoint -> {checkpoint_path}")
            refit_payload.update(
                {
                    "performed": True,
                    "from_scratch": True,
                    "epochs": refit_epochs,
                    "combined_train_size": int(len(refit_df)),
                    "status": "completed",
                }
            )
        else:
            torch.save(model.state_dict(), checkpoint_path)
            refit_payload.update(
                {
                    "performed": False,
                    "from_scratch": False,
                    "epochs": 0,
                    "combined_train_size": int(len(train_df) + len(val_df)),
                    "status": "disabled_by_config",
                }
            )
    else:
        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=cfg.device)
            model.load_state_dict(state)
        selected_val_payload = {
            "evaluation_split": "disabled",
            "selection_score": None,
            "quality": {},
            "consistency": {},
        }
        if selection_based_on != "train_loss_threshold":
            selection_based_on = "last_epoch_train_loss"

    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=cfg.device)
        model.load_state_dict(state)

    runtime_sec = float(time.time() - start_time)
    _, _, _, _, final_payload = evaluate(
        model,
        test_dl,
        cfg,
        proc,
        print_examples=cfg.use_refinement,
        num_examples=10,
        persist=True,
        extra_metadata={
            "best_epoch": best_epoch,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "runtime_sec": runtime_sec,
            "split_file": cfg.split_file,
            "split_seed": cfg.split_seed,
            "seed": cfg.seed,
            "run_name": cfg.run_name,
            "group_source": test_ds.group_source,
            "evaluation_split": "test",
            "train_loss_stop_threshold": train_loss_stop_threshold,
            "stopped_on_train_loss_threshold": stopped_on_train_loss_threshold,
            "selection": {
                "metric": cfg.selection_metric if has_validation_split else "train_loss",
                "based_on": selection_based_on,
                "best_epoch": best_epoch,
                "best_score": None if not has_validation_split else best_score,
                "best_train_loss": best_train_loss,
                "last_train_loss": last_train_loss,
                "weights": {"w_q": cfg.w_q, "w_c": cfg.w_c},
                "refit_trainval": bool(refit_payload["performed"]),
                "train_loss_stop_threshold": train_loss_stop_threshold,
                "stopped_on_train_loss_threshold": stopped_on_train_loss_threshold,
            },
            "validation_metrics": {
                "selection_score": (
                    float(selected_val_payload["selection_score"])
                    if selected_val_payload.get("selection_score") is not None else None
                ),
                "quality": selected_val_payload["quality"],
                "consistency": selected_val_payload["consistency"],
            },
            "refit": refit_payload,
            "split_role_counts": {
                "train": len(train_df),
                "validation": len(val_df),
                "test": len(test_df),
            },
            "split_details": split_payload,
        },
        return_payload=True,
    )
    final_payload["test_selection_score"] = float(final_payload.get("selection_score", 0.0))
    final_payload["selection_score"] = (
        float(best_score) if has_validation_split else float(final_payload["test_selection_score"])
    )

    train_log = {
        "run_name": cfg.run_name,
        "seed": cfg.seed,
        "split_seed": cfg.split_seed,
        "split_file": cfg.split_file,
        "best_epoch": best_epoch,
        "best_selection_score": float(best_score) if has_validation_split else None,
        "best_train_loss": best_train_loss,
        "last_train_loss": last_train_loss,
        "train_loss_stop_threshold": train_loss_stop_threshold,
        "stopped_on_train_loss_threshold": stopped_on_train_loss_threshold,
        "evaluation_split": "test",
        "selection_split": "validation" if has_validation_split else "disabled",
        "selected_validation_metrics": selected_val_payload,
        "history": history,
        "refit_on_trainval": bool(refit_payload["performed"]),
        "refit": refit_payload,
        "refit_history": refit_history,
        "split": split_payload,
    }
    with open(cfg.save_train_log_json, "w", encoding="utf-8") as f:
        json.dump(train_log, f, ensure_ascii=False, indent=2)

    print("\nTraining Complete!")
    if has_validation_split:
        print(f"Best selection score: {best_score:.4f} (epoch {best_epoch})")
        if refit_payload["performed"]:
            print(
                f"Refit on train+validation: enabled, epochs={refit_payload['epochs']}, "
                f"combined_train_size={refit_payload['combined_train_size']}"
            )
        else:
            print("Refit on train+validation: disabled")
    else:
        if stopped_on_train_loss_threshold:
            print(
                f"Train-loss early stop triggered at epoch {best_epoch} "
                f"(threshold={train_loss_stop_threshold:.6f})"
            )
        else:
            print(f"Final epoch used for evaluation: epoch {best_epoch} (validation disabled)")

    return {
        "run_dir": run_dir,
        "checkpoint_path": checkpoint_path,
        "preds_path": cfg.save_val_preds_csv,
        "metrics_path": cfg.save_val_metrics_json,
        "train_log_path": cfg.save_train_log_json,
        "config_path": cfg.save_config_json,
        "metrics": final_payload,
        "split": split_payload,
    }


def main():
    cfg = TrainingConfig()
    parser = argparse.ArgumentParser("Baseline CLIP eval")
    parser.add_argument('--data_csv_path');
    parser.add_argument('--image_base_dir')
    parser.add_argument('--clip_model_name', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--epochs', type=int);
    parser.add_argument('--batch_size', type=int);
    parser.add_argument('--lr', type=float)
    parser.add_argument('--w_q', type=float);
    parser.add_argument('--w_c', type=float);
    parser.add_argument('--seed', type=int)
    parser.add_argument('--test_size', type=float)
    parser.add_argument('--val_size_within_train', type=float)
    parser.add_argument('--disable_validation_split', action='store_true')
    parser.add_argument('--train_loss_stop_threshold', type=float, default=5e-4,
                        help='Early stop when epoch-average total train loss falls below this threshold')
    parser.add_argument('--split_seed', type=int)
    parser.add_argument('--split_file', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--group_column', type=str)
    parser.add_argument('--selection_metric', choices=['avg_srocc', 'weighted_qc_srocc', 'consistency_srocc', 'quality_srocc'])
    parser.add_argument('--pure_linear_probe', action='store_true')
    parser.add_argument('--no_train_aug', action='store_true',
                        help='Disable train-time augmentation (RandomResizedCrop/RandomHorizontalFlip)')
    parser.add_argument('--crop_scale_min', type=float,
                        help='RandomResizedCrop minimum scale (default: 0.8)')
    parser.add_argument('--hflip_p', type=float,
                        help='RandomHorizontalFlip probability (default: 0.5)')
    parser.add_argument('--use_hetero_weight', action='store_true',
                        help='Use heteroscedastic weighting with std_quality/std_align')
    parser.add_argument('--std_floor', type=float,
                        help='Variance floor for heteroscedastic weighting (default: 1e-3)')
    parser.add_argument('--hetero_weight_clip', type=float,
                        help='Max per-sample hetero weight after 1/var (default: 25.0)')
    parser.add_argument('--use_group_dro', action='store_true',
                        help='Enable group-DRO using generator groups from image name prefix (disabled by default)')
    parser.add_argument('--no_group_dro', action='store_true',
                        help='Disable group-DRO explicitly')
    parser.add_argument('--group_dro_temp', type=float,
                        help='Softmax temperature for group-DRO (default: 1.0)')
    parser.add_argument('--group_dro_lambda', type=float,
                        help='Mixing weight in [0,1] for group-DRO vs mean loss (default: 0.0)')
    parser.add_argument('--freeze_clip', action='store_true')
    parser.add_argument('--use_explanations', action='store_true')
    parser.add_argument('--use_two_branch', action='store_true')
    parser.add_argument('--w_exp', type=float)
    parser.add_argument('--explanation_column')
    parser.add_argument('--use_refinement', action='store_true', help='Enable two-stage consistency refinement')
    parser.add_argument('--refinement_layers', type=int, help='Number of transformer layers in refinement')
    parser.add_argument('--refinement_heads', type=int, help='Number of attention heads in refinement')
    parser.add_argument('--refinement_dim', type=int, help='Hidden dimension for refinement module')
    parser.add_argument('--no_strict_residual', action='store_true',
                        help='Disable strict residual learning (default: enabled)')

    # ===== residual learning =====
    parser.add_argument('--no_residual_learning', action='store_true',
                        help='Disable residual learning (enabled by default)')
    parser.add_argument('--residual_scale_q', type=float,
                        help='Quality residual scale (default: 0.2)')
    parser.add_argument('--residual_scale_c', type=float,
                        help='Consistency residual scale (default: 0.2)')
    parser.add_argument('--partial_freeze', action='store_true',
                        help='Enable partial CLIP freezing (finetune later layers only)')
    parser.add_argument('--freeze_layers', type=int,
                        help='Freeze first N transformer layers (default: 8)')
    parser.add_argument('--no_pcrc', action='store_true',
                        help='Disable Prompt-Conditioned Residual Calibration (enabled by default)')
    parser.add_argument('--no_film', action='store_true',
                        help='Disable FiLM modulation inside residual calibration')
    parser.add_argument('--use_logic_concat', action='store_true',
                        help='Use direct [CLIP features ; logic vector] concat MLP instead of FiLM when PCRC is disabled')
    parser.add_argument('--pcrc_num_anchors', type=int,
                        help='Number of prompt anchors A (default: 16)')
    parser.add_argument('--pcrc_hidden', type=int,
                        help='Hidden size for PCRC residual head (default: 256)')
    parser.add_argument('--pcrc_learnable_anchors', action='store_true',
                        help='Use learnable anchor embeddings instead of frozen anchors')
    parser.add_argument('--pcrc_dynamic_anchors', action='store_true',
                        help='Recompute frozen anchor embeddings every forward (for full CLIP finetuning)')
    parser.add_argument('--pcrc_anchor_texts', type=str,
                        help='Comma-separated custom anchor prompts')

    # ===== RACL =====
    parser.add_argument('--use_rank_loss', action='store_true',
                        help='Enable ranking loss for direct correlation optimization')
    parser.add_argument('--rank_alpha', type=float,
                        help='Ranking logits slope alpha (default: 10.0)')
    parser.add_argument('--rank_pairs', type=int,
                        help='Sampled pair count per batch (default: 64)')
    parser.add_argument('--rank_lambda', type=float,
                        help='Ranking loss weight (default: 0.5)')
    parser.add_argument('--run_smoke_tests', action='store_true',
                        help='Run module smoke tests and exit')

    # PromptMHA
    parser.add_argument('--no_prompt_mha', action='store_true')
    parser.add_argument('--prompt_mha_heads', type=int)
    parser.add_argument('--prompt_mha_dropout', type=float)

    # MoE
    parser.add_argument('--use_moe', action='store_true')
    parser.add_argument('--moe_num_experts', type=int)
    parser.add_argument('--moe_gate_hidden', type=int)
    parser.add_argument('--moe_tau', type=float)
    parser.add_argument('--moe_entropy_lambda', type=float)

    # LoRA
    parser.add_argument('--no_lora', action='store_true')
    parser.add_argument('--lora_r', type=int)
    parser.add_argument('--lora_alpha', type=int)
    parser.add_argument('--lora_dropout', type=float)
    parser.add_argument('--lora_target_modules', type=str)

    # group-DRO
    parser.add_argument('--group_dro_mode', choices=['softmax_batch', 'expgrad'])
    parser.add_argument('--group_dro_eta', type=float)

    # Eval/export
    parser.add_argument('--save_val_preds_csv', type=str)
    parser.add_argument('--save_token_importance_jsonl', type=str)
    parser.add_argument('--token_topk', type=int)
    parser.add_argument('--bootstrap_iters', type=int)
    parser.add_argument('--bootstrap_seed', type=int)
    parser.add_argument('--save_val_metrics_json', type=str)
    parser.add_argument('--save_train_log_json', type=str)
    parser.add_argument('--save_config_json', type=str)
    parser.add_argument('--save_checkpoint_name', type=str)
    parser.add_argument('--group_metric_min_size', type=int)
    parser.add_argument('--save_ig_topk', action='store_true')
    parser.add_argument('--ig_steps', type=int)
    parser.add_argument('--ig_max_batches', type=int)
    parser.add_argument('--funnel_cache_jsonl', type=str, default="")
    parser.add_argument('--text_source', choices=TEXT_SOURCE_CHOICES, default="raw_prompt")
    parser.add_argument('--no_focus_local_branch', action='store_true',
                        help='Disable the focus-image local CLIP consistency branch')
    parser.add_argument('--focus_local_scale', type=float,
                        help='Initial prior for the learnable focus-local blending gate in [0,1]')
    parser.add_argument('--focus_local_text_source', choices=TEXT_SOURCE_CHOICES, default="funnel_selected_prompt",
                        help='Text source used with the focus image for local CLIP scoring')
    parser.add_argument('--no_focus_local_fallback_to_global', action='store_true',
                        help='Do not fall back to the full prompt when focus-local text is missing')
    parser.add_argument('--no_refit_trainval', action='store_true',
                        help='Skip retraining from scratch on train+validation after selecting best epoch on validation')

    args = parser.parse_args()
    if args.data_csv_path: cfg.data_csv_path = args.data_csv_path
    if args.image_base_dir: cfg.image_base_dir = args.image_base_dir
    if args.clip_model_name: cfg.clip_model_name = args.clip_model_name
    if args.device is not None: cfg.device = torch.device(args.device)
    if args.epochs: cfg.epochs = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.lr: cfg.lr = args.lr
    if args.w_q: cfg.w_q = args.w_q
    if args.w_c: cfg.w_c = args.w_c
    if args.seed is not None: cfg.seed = args.seed
    if args.test_size is not None: cfg.test_size = args.test_size
    # train.py entrypoint now runs in train/test mode only: no validation split, no train+val refit.
    cfg.disable_validation_split = True
    cfg.refit_on_trainval = False
    cfg.train_loss_stop_threshold = args.train_loss_stop_threshold
    if args.val_size_within_train is not None: cfg.val_size_within_train = args.val_size_within_train
    if args.disable_validation_split: cfg.disable_validation_split = True
    if args.split_seed is not None: cfg.split_seed = args.split_seed
    if args.split_file is not None: cfg.split_file = args.split_file
    if args.output_dir is not None: cfg.output_dir = args.output_dir
    if args.run_name is not None: cfg.run_name = args.run_name
    if args.group_column is not None: cfg.group_column = args.group_column
    if args.selection_metric is not None: cfg.selection_metric = args.selection_metric
    if args.pure_linear_probe: cfg.pure_linear_probe = True
    if args.no_train_aug: cfg.use_train_aug = False
    if args.crop_scale_min is not None: cfg.crop_scale_min = args.crop_scale_min
    if args.hflip_p is not None: cfg.hflip_p = args.hflip_p
    if args.use_hetero_weight: cfg.use_hetero_weight = True
    if args.std_floor is not None: cfg.std_floor = args.std_floor
    if args.hetero_weight_clip is not None: cfg.hetero_weight_clip = args.hetero_weight_clip
    if args.no_group_dro: cfg.use_group_dro = False
    if args.use_group_dro: cfg.use_group_dro = True
    if args.group_dro_temp is not None: cfg.group_dro_temp = args.group_dro_temp
    if args.group_dro_lambda is not None: cfg.group_dro_lambda = args.group_dro_lambda
    cfg.group_dro_lambda = max(0.0, min(1.0, cfg.group_dro_lambda))
    if args.freeze_clip: cfg.freeze_clip = True
    if args.use_explanations: cfg.use_explanations = True
    if args.w_exp is not None: cfg.w_exp = args.w_exp
    if args.explanation_column: cfg.explanation_column = args.explanation_column
    if args.use_refinement: cfg.use_refinement = True
    if args.refinement_layers: cfg.refinement_layers = args.refinement_layers
    if args.refinement_heads: cfg.refinement_heads = args.refinement_heads
    if args.refinement_dim: cfg.refinement_dim = args.refinement_dim
    if args.no_strict_residual: cfg.strict_residual = False

    # residual learning config
    if args.no_residual_learning: cfg.use_residual_learning = False
    if args.residual_scale_q is not None: cfg.residual_scale_q = args.residual_scale_q
    if args.residual_scale_c is not None: cfg.residual_scale_c = args.residual_scale_c
    if args.partial_freeze: cfg.partial_freeze = True
    if args.freeze_layers: cfg.freeze_layers = args.freeze_layers
    if args.no_pcrc: cfg.use_pcrc = False
    if args.no_film: cfg.use_film = False
    if args.use_logic_concat: cfg.use_logic_concat = True
    if cfg.use_logic_concat and (not cfg.use_pcrc) and cfg.use_residual_learning:
        cfg.use_film = False
    if args.pcrc_num_anchors is not None: cfg.pcrc_num_anchors = args.pcrc_num_anchors
    if args.pcrc_hidden is not None: cfg.pcrc_hidden = args.pcrc_hidden
    if args.pcrc_learnable_anchors: cfg.pcrc_learnable_anchors = True
    if args.pcrc_dynamic_anchors: cfg.pcrc_dynamic_anchors = True
    if args.pcrc_anchor_texts: cfg.pcrc_anchor_texts = args.pcrc_anchor_texts
    if args.use_rank_loss: cfg.use_rank_loss = True

    # RACL config
    if args.rank_alpha is not None: cfg.rank_alpha = args.rank_alpha
    if args.rank_pairs is not None: cfg.rank_pairs = args.rank_pairs
    if args.rank_lambda is not None: cfg.rank_lambda = args.rank_lambda

    if args.no_prompt_mha: cfg.use_prompt_mha = False
    if args.prompt_mha_heads is not None: cfg.prompt_mha_heads = args.prompt_mha_heads
    if args.prompt_mha_dropout is not None: cfg.prompt_mha_dropout = args.prompt_mha_dropout

    if args.use_moe: cfg.use_moe = True
    if args.moe_num_experts is not None: cfg.moe_num_experts = args.moe_num_experts
    if args.moe_gate_hidden is not None: cfg.moe_gate_hidden = args.moe_gate_hidden
    if args.moe_tau is not None: cfg.moe_tau = args.moe_tau
    if args.moe_entropy_lambda is not None: cfg.moe_entropy_lambda = args.moe_entropy_lambda

    if args.no_lora: cfg.use_lora = False
    if args.lora_r is not None: cfg.lora_r = args.lora_r
    if args.lora_alpha is not None: cfg.lora_alpha = args.lora_alpha
    if args.lora_dropout is not None: cfg.lora_dropout = args.lora_dropout
    if args.lora_target_modules is not None: cfg.lora_target_modules = args.lora_target_modules

    if args.group_dro_mode is not None: cfg.group_dro_mode = args.group_dro_mode
    if args.group_dro_eta is not None: cfg.group_dro_eta = args.group_dro_eta

    if args.save_val_preds_csv is not None: cfg.save_val_preds_csv = args.save_val_preds_csv
    if args.save_token_importance_jsonl is not None: cfg.save_token_importance_jsonl = args.save_token_importance_jsonl
    if args.token_topk is not None: cfg.token_topk = args.token_topk
    if args.bootstrap_iters is not None: cfg.bootstrap_iters = args.bootstrap_iters
    if args.bootstrap_seed is not None: cfg.bootstrap_seed = args.bootstrap_seed
    if args.save_val_metrics_json is not None: cfg.save_val_metrics_json = args.save_val_metrics_json
    if args.save_train_log_json is not None: cfg.save_train_log_json = args.save_train_log_json
    if args.save_config_json is not None: cfg.save_config_json = args.save_config_json
    if args.save_checkpoint_name is not None: cfg.save_checkpoint_name = args.save_checkpoint_name
    if args.group_metric_min_size is not None: cfg.group_metric_min_size = args.group_metric_min_size
    if args.save_ig_topk: cfg.save_ig_topk = True
    if args.ig_steps is not None: cfg.ig_steps = args.ig_steps
    if args.ig_max_batches is not None: cfg.ig_max_batches = args.ig_max_batches
    if args.funnel_cache_jsonl is not None: cfg.funnel_cache_jsonl = args.funnel_cache_jsonl
    if args.text_source is not None: cfg.text_source = args.text_source
    if args.no_focus_local_branch: cfg.use_focus_local_branch = False
    if args.focus_local_scale is not None: cfg.focus_local_scale = args.focus_local_scale
    if args.focus_local_text_source is not None: cfg.focus_local_text_source = args.focus_local_text_source
    if args.no_focus_local_fallback_to_global: cfg.focus_local_fallback_to_global = False
    if args.no_refit_trainval: cfg.refit_on_trainval = False

    if args.run_smoke_tests:
        run_module_smoke_tests()
        return

    run_training(cfg)
    return

    df = pd.read_csv(cfg.data_csv_path)
    if args.funnel_cache_jsonl:
        df = merge_funnel_cache(df, args.funnel_cache_jsonl)
    train_df, val_df = train_test_split(df, test_size=cfg.test_size, random_state=42)
    proc = CLIPProcessor.from_pretrained(cfg.clip_model_name)
    train_tfms = []
    if cfg.use_train_aug:
        train_tfms.extend([
            transforms.RandomResizedCrop(cfg.image_size, scale=(cfg.crop_scale_min, 1.0)),
            transforms.RandomHorizontalFlip(p=cfg.hflip_p),
        ])
    else:
        train_tfms.append(transforms.Resize((cfg.image_size, cfg.image_size)))
    train_tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(proc.image_processor.image_mean, proc.image_processor.image_std),
    ])
    tf_train = transforms.Compose(train_tfms)
    tf_val = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(proc.image_processor.image_mean, proc.image_processor.image_std)
    ])
    # If a custom explanation column is provided, ensure it exists (optional)
    if cfg.explanation_column != "explanation" and cfg.explanation_column in train_df.columns:
        train_df = train_df.rename(columns={cfg.explanation_column: "explanation"})
        val_df = val_df.rename(columns={cfg.explanation_column: "explanation"})
    train_ds = BaselineDataset(train_df, cfg.image_base_dir, proc, tf_train)
    val_ds = BaselineDataset(val_df, cfg.image_base_dir, proc, tf_val)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    dro_state = None
    if cfg.use_group_dro and cfg.group_dro_mode == "expgrad":
        dro_state = GroupDROState(num_groups=len(train_ds.group2id), eta=cfg.group_dro_eta, device=cfg.device)

    # Build refinement config
    refinement_cfg = {
        'hidden_dim': cfg.refinement_dim,
        'num_layers': cfg.refinement_layers,
        'num_heads': cfg.refinement_heads
    } if cfg.use_refinement else None

    pcrc_anchor_texts = None
    if cfg.use_residual_learning and cfg.use_pcrc:
        pcrc_anchor_texts = build_pcrc_anchor_texts(
            num_anchors=cfg.pcrc_num_anchors,
            anchor_texts_csv=cfg.pcrc_anchor_texts,
        )
    lora_target_modules = _parse_target_modules(cfg.lora_target_modules)

    model = BaselineCLIPScore(
        cfg.clip_model_name,
        freeze=cfg.freeze_clip,
        use_refinement=cfg.use_refinement,
        refinement_cfg=refinement_cfg,
        use_two_branch=cfg.use_two_branch,
        use_residual_learning=cfg.use_residual_learning,
        residual_scale_q=cfg.residual_scale_q,
        residual_scale_c=cfg.residual_scale_c,
        partial_freeze=cfg.partial_freeze,
        freeze_layers=cfg.freeze_layers,
        use_pcrc=cfg.use_pcrc,
        pcrc_hidden=cfg.pcrc_hidden,
        pcrc_anchor_texts=pcrc_anchor_texts,
        pcrc_learnable_anchors=cfg.pcrc_learnable_anchors,
        pcrc_dynamic_anchors=cfg.pcrc_dynamic_anchors,
        clip_processor=proc,
        use_prompt_mha=cfg.use_prompt_mha,
        prompt_mha_heads=cfg.prompt_mha_heads,
        prompt_mha_dropout=cfg.prompt_mha_dropout,
        use_moe=cfg.use_moe,
        moe_num_experts=cfg.moe_num_experts,
        moe_gate_hidden=cfg.moe_gate_hidden,
        moe_tau=cfg.moe_tau,
        use_lora=cfg.use_lora,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_target_modules=lora_target_modules,
    ).to(cfg.device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * len(train_dl) * cfg.epochs), len(train_dl) * cfg.epochs)

    # Print config
    print(f"\n{'=' * 70}")
    print(f"Training Configuration:")
    print(f"  - Epochs: {cfg.epochs}, Batch Size: {cfg.batch_size}, LR: {cfg.lr}")
    print(f"  - Loss Weights: w_q={cfg.w_q}, w_c={cfg.w_c}")
    print(f"  - Use Explanations: {cfg.use_explanations}" + (f" (w_exp={cfg.w_exp})" if cfg.use_explanations else ""))
    print(f"  - Train Aug: {cfg.use_train_aug} (crop_scale_min={cfg.crop_scale_min}, hflip_p={cfg.hflip_p})")
    print(f"  - Hetero Weight: {cfg.use_hetero_weight} (std_floor={cfg.std_floor}, clip={cfg.hetero_weight_clip})")
    print(f"  - Group DRO: {cfg.use_group_dro} (mode={cfg.group_dro_mode}, temp={cfg.group_dro_temp}, lambda={cfg.group_dro_lambda}, eta={cfg.group_dro_eta})")
    print(f"  - PromptMHA: {cfg.use_prompt_mha} (heads={cfg.prompt_mha_heads}, dropout={cfg.prompt_mha_dropout})")
    print(f"  - MoE: {cfg.use_moe} (experts={cfg.moe_num_experts}, tau={cfg.moe_tau}, ent_lambda={cfg.moe_entropy_lambda})")
    print(f"  - LoRA: {cfg.use_lora} (r={cfg.lora_r}, alpha={cfg.lora_alpha}, dropout={cfg.lora_dropout})")
    print(f"  - Text Source: {cfg.text_source}")
    print(f"  - Export: preds={cfg.save_val_preds_csv}, metrics={cfg.save_val_metrics_json}, token={cfg.save_token_importance_jsonl}, ig={cfg.save_ig_topk}")
    print("")
    print("  CLIP Freeze Strategy:")
    if cfg.use_lora:
        print("     - LoRA PEFT (base CLIP frozen)")
    elif cfg.freeze_clip:
        print("     - Frozen CLIP (linear probing)")
    elif cfg.partial_freeze:
        print(f"     - Partial freeze: first {cfg.freeze_layers} layers")
    else:
        print("     - Full CLIP finetuning (end-to-end)")

    print("")
    print("  Prediction Architecture:")
    if cfg.use_residual_learning:
        print("     - Residual learning enabled")
        print(f"       q = clip(q_base + lambda_q * delta_q, 0, 1), lambda_q={cfg.residual_scale_q}")
        print(f"       c = clip(c_base + lambda_c * delta_c, 0, 1), lambda_c={cfg.residual_scale_c}")
        print("       c_base = (cos(img, txt) + 1) / 2")
        if cfg.use_pcrc:
            dim = model.clip.config.projection_dim
            a = len(pcrc_anchor_texts) if pcrc_anchor_texts is not None else cfg.pcrc_num_anchors
            h = cfg.pcrc_hidden
            pcrc_params = (3 * dim + a) * h + h + (h * 2) + 2
            print(f"       PCRC: enabled (A={a}, hidden={h}, learnable_anchors={cfg.pcrc_learnable_anchors}, dynamic_anchors={cfg.pcrc_dynamic_anchors})")
            print("       (delta_q, delta_c) = r(v, u, p(x)), p(x)=[<v,u_1>,...,<v,u_A>]")
            print(f"       PCRC MLP params (approx): {pcrc_params}")
        else:
            print("       PCRC: disabled (fallback residual heads)")
    else:
        print("     - Direct prediction mode")
        if cfg.use_refinement:
            residual_mode = "Strict Residual" if cfg.strict_residual else "Standard"
            print(f"       Refinement: {cfg.refinement_layers} layers, {cfg.refinement_heads} heads, dim={cfg.refinement_dim}, mode={residual_mode}")

    if cfg.use_rank_loss:
        print("")
        print("  RACL: enabled")
        print(f"     - pairs per batch: {cfg.rank_pairs}")
        print(f"     - alpha: {cfg.rank_alpha}")
        print(f"     - lambda: {cfg.rank_lambda}")
    else:
        print("")
        print("  RACL: disabled (use --use_rank_loss to enable)")

    print(f"{'=' * 70}\n")

    best_sc = -1
    for ep in range(cfg.epochs):
        train_loss, lq, lc, le, lrq, lrc = train_epoch(model, train_dl, opt, sched, cfg, dro_state=dro_state)

        # Print examples every 5 epochs or at the last epoch
        print_examples = cfg.use_refinement and ((ep + 1) % 5 == 0 or (ep + 1) == cfg.epochs)
        s_q, p_q, s_c, p_c = evaluate(model, val_dl, cfg, proc, print_examples=print_examples, num_examples=5)

        if cfg.use_rank_loss:
            if cfg.use_explanations:
                print(
                    f"Ep{ep + 1} TrainLoss={train_loss:.4f}(Q{lq:.4f},C{lc:.4f},E{le:.4f},RankQ{lrq:.4f},RankC{lrc:.4f})  Val SROCC_Q={s_q:.4f},SROCC_C={s_c:.4f}")
            else:
                print(
                    f"Ep{ep + 1} TrainLoss={train_loss:.4f}(Q{lq:.4f},C{lc:.4f},RankQ{lrq:.4f},RankC{lrc:.4f})  Val SROCC_Q={s_q:.4f},SROCC_C={s_c:.4f}")
        else:
            if cfg.use_explanations:
                print(
                    f"Ep{ep + 1} TrainLoss={train_loss:.4f}(Q{lq:.4f},C{lc:.4f},E{le:.4f})  Val SROCC_Q={s_q:.4f},SROCC_C={s_c:.4f}")
            else:
                print(
                    f"Ep{ep + 1} TrainLoss={train_loss:.4f}(Q{lq:.4f},C{lc:.4f})  Val SROCC_Q={s_q:.4f},SROCC_C={s_c:.4f}")

        if s_c > best_sc:
            best_sc = s_c
            # Save the current best checkpoint according to validation alignment SROCC.
            if cfg.use_rank_loss:
                save_name = f"baseline_racl_lambda{cfg.rank_lambda}_best.pt"
            elif cfg.use_residual_learning:
                if cfg.use_pcrc:
                    save_name = f"baseline_pcrc_A{cfg.pcrc_num_anchors}_best.pt"
                elif cfg.partial_freeze:
                    save_name = f"baseline_residual_partial_freeze_{cfg.freeze_layers}L_best.pt"
                else:
                    save_name = "baseline_residual_best.pt"
            elif cfg.use_refinement:
                save_name = "baseline_refinement_best.pt"
            else:
                save_name = "baseline_best.pt"
            torch.save(model.state_dict(), save_name)
            print(f"[Checkpoint] New best SROCC_C={s_c:.4f} -> {save_name}")

    # Final evaluation with examples
    if cfg.use_refinement:
        print("\n" + "=" * 70)
        print("FINAL EVALUATION WITH EXAMPLES")
        print("=" * 70)
        s_q, p_q, s_c, p_c = evaluate(model, val_dl, cfg, proc, print_examples=True, num_examples=10)

    print("\nTraining Complete!")
    print(f"Best SROCC_C: {best_sc:.4f}")


if __name__ == "__main__":
    main()
