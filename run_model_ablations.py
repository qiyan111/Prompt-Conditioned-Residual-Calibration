#!/usr/bin/env python3
"""Run ONSRC ablations under multi-split, multi-seed evaluation."""

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import train as model_mod


DEFAULT_SPLIT_SEEDS = [42, 52, 62]
DEFAULT_TRAIN_SEEDS = [11, 22, 33]
REFERENCE_VARIANT = "residual_best_ref"
BASELINE_CANDIDATE_VARIANTS = ["direct_with_funnel", "nofunnel_direct", "nofunnel_linear"]
COMPARISON_SPECS = [
    {
        "name": "onsrc_vs_strongest_fair_baseline",
        "mode": "auto_strongest_baseline",
        "candidate_variant": REFERENCE_VARIANT,
        "label": "ONSRC vs strongest fair rerun baseline",
    },
    {
        "name": "full_vs_no_funnel",
        "mode": "fixed",
        "reference_variant": "residual_no_funnel",
        "candidate_variant": REFERENCE_VARIANT,
        "label": "Full model vs w/o Funnel",
    },
    {
        "name": "full_vs_no_film",
        "mode": "fixed",
        "reference_variant": "residual_no_film",
        "candidate_variant": REFERENCE_VARIANT,
        "label": "Full model vs w/o FiLM",
    },
    {
        "name": "full_vs_no_prompt_mha",
        "mode": "fixed",
        "reference_variant": "residual_no_prompt_mha",
        "candidate_variant": REFERENCE_VARIANT,
        "label": "Full model vs w/o PromptMHA",
    },
    {
        "name": "full_vs_no_pcrc",
        "mode": "fixed",
        "reference_variant": "residual_no_pcrc",
        "candidate_variant": REFERENCE_VARIANT,
        "label": "Full model vs w/o PCRC",
    },
    {
        "name": "full_vs_no_focus_local",
        "mode": "fixed",
        "reference_variant": "residual_no_focus_local",
        "candidate_variant": REFERENCE_VARIANT,
        "label": "Full model vs w/o Focus-Local",
    },
]
CONFIG_MATCH_KEYS = [
    "data_csv_path",
    "image_base_dir",
    "clip_model_name",
    "epochs",
    "batch_size",
    "lr",
    "w_q",
    "w_c",
    "test_size",
    "val_size_within_train",
    "seed",
    "split_seed",
    "split_file",
    "disable_validation_split",
    "run_name",
    "group_column",
    "selection_metric",
    "freeze_clip",
    "pure_linear_probe",
    "text_source",
    "use_two_branch",
    "use_refinement",
    "use_residual_learning",
    "use_group_dro",
    "group_dro_lambda",
    "use_pcrc",
    "use_film",
    "use_prompt_mha",
    "prompt_mha_heads",
    "prompt_mha_dropout",
    "use_lora",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_target_modules",
    "use_focus_local_branch",
    "focus_local_scale",
    "focus_local_text_source",
    "focus_local_fallback_to_global",
    "funnel_cache_jsonl",
    "refit_on_trainval",
    "group_metric_min_size",
]
FOCUS_METADATA_KEYS = [
    "use_focus_local_branch",
    "focus_local_scale",
    "focus_local_text_source",
    "focus_local_fallback_to_global",
]

DISPLAY_NAME_BY_VARIANT = {
    "residual_best_ref": "ONSRC (Full)",
    "direct_with_funnel": "Direct Prediction (with Funnel)",
    "residual_no_funnel": "ONSRC w/o Funnel",
    "residual_no_film": "ONSRC w/o FiLM",
    "residual_no_prompt_mha": "ONSRC w/o PromptMHA",
    "residual_no_pcrc": "ONSRC w/o PCRC",
    "residual_no_focus_local": "ONSRC w/o Focus-Local",
    "nofunnel_direct": "w/o Funnel + Direct Prediction",
    "nofunnel_linear": "w/o Funnel + Linear Probe",
}


def build_ablation_variants() -> List[Dict[str, Any]]:
    return [
        {
            "name": "residual_best_ref",
            "overrides": {
                "use_residual_learning": True,
                "use_pcrc": True,
                "use_film": True,
                "use_prompt_mha": True,
                "use_lora": True,
                "use_focus_local_branch": True,
                "focus_local_scale": 1.0,
                "focus_local_text_source": "funnel_selected_prompt",
                "focus_local_fallback_to_global": True,
                "use_group_dro": False,
                "group_dro_lambda": 0.0,
                "use_refinement": False,
                "text_source": "raw_prompt",
            },
        },
        {
            "name": "direct_with_funnel",
            "overrides": {
                "use_residual_learning": False,
                "use_pcrc": False,
                "use_lora": True,
                "use_focus_local_branch": True,
                "focus_local_scale": 1.0,
                "focus_local_text_source": "funnel_selected_prompt",
                "focus_local_fallback_to_global": True,
                "use_group_dro": False,
                "group_dro_lambda": 0.0,
                "use_refinement": False,
                "text_source": "raw_prompt",
            },
        },
        {
            "name": "residual_no_funnel",
            "overrides": {
                "use_residual_learning": True,
                "use_pcrc": True,
                "use_film": True,
                "use_prompt_mha": True,
                "use_lora": True,
                "use_focus_local_branch": False,
                "focus_local_scale": 1.0,
                "focus_local_text_source": "funnel_selected_prompt",
                "focus_local_fallback_to_global": True,
                "use_group_dro": False,
                "group_dro_lambda": 0.0,
                "use_refinement": False,
                "funnel_cache_jsonl": "",
                "text_source": "raw_prompt",
            },
        },
        {
            "name": "residual_no_film",
            "overrides": {
                "use_residual_learning": True,
                "use_pcrc": True,
                "use_film": False,
                "use_prompt_mha": True,
                "use_lora": True,
                "use_focus_local_branch": True,
                "focus_local_scale": 1.0,
                "focus_local_text_source": "funnel_selected_prompt",
                "focus_local_fallback_to_global": True,
                "use_group_dro": False,
                "group_dro_lambda": 0.0,
                "use_refinement": False,
                "text_source": "raw_prompt",
            },
        },
        {
            "name": "residual_no_prompt_mha",
            "overrides": {
                "use_residual_learning": True,
                "use_pcrc": True,
                "use_film": True,
                "use_prompt_mha": False,
                "use_lora": True,
                "use_focus_local_branch": True,
                "focus_local_scale": 1.0,
                "focus_local_text_source": "funnel_selected_prompt",
                "focus_local_fallback_to_global": True,
                "use_group_dro": False,
                "group_dro_lambda": 0.0,
                "use_refinement": False,
                "text_source": "raw_prompt",
            },
        },
        {
            "name": "residual_no_pcrc",
            "overrides": {
                "use_residual_learning": True,
                "use_pcrc": False,
                "use_film": True,
                "use_prompt_mha": True,
                "use_lora": True,
                "use_focus_local_branch": True,
                "focus_local_scale": 1.0,
                "focus_local_text_source": "funnel_selected_prompt",
                "focus_local_fallback_to_global": True,
                "use_group_dro": False,
                "group_dro_lambda": 0.0,
                "use_refinement": False,
                "text_source": "raw_prompt",
            },
        },
        {
            "name": "residual_no_focus_local",
            "overrides": {
                "use_residual_learning": True,
                "use_pcrc": True,
                "use_film": True,
                "use_prompt_mha": True,
                "use_lora": True,
                "use_focus_local_branch": False,
                "focus_local_scale": 1.0,
                "focus_local_text_source": "funnel_selected_prompt",
                "focus_local_fallback_to_global": True,
                "use_group_dro": False,
                "group_dro_lambda": 0.0,
                "use_refinement": False,
                "text_source": "raw_prompt",
            },
        },
        {
            "name": "nofunnel_direct",
            "overrides": {
                "use_residual_learning": False,
                "use_pcrc": False,
                "use_lora": True,
                "use_focus_local_branch": False,
                "focus_local_scale": 1.0,
                "focus_local_text_source": "funnel_selected_prompt",
                "focus_local_fallback_to_global": True,
                "use_group_dro": False,
                "group_dro_lambda": 0.0,
                "use_refinement": False,
                "funnel_cache_jsonl": "",
                "text_source": "raw_prompt",
            },
        },
        {
            "name": "nofunnel_linear",
            "overrides": {
                "pure_linear_probe": True,
                "freeze_clip": True,
                "use_lora": False,
                "use_residual_learning": False,
                "use_pcrc": False,
                "use_focus_local_branch": False,
                "focus_local_scale": 1.0,
                "focus_local_text_source": "funnel_selected_prompt",
                "focus_local_fallback_to_global": True,
                "use_group_dro": False,
                "group_dro_lambda": 0.0,
                "use_refinement": False,
                "funnel_cache_jsonl": "",
                "text_source": "raw_prompt",
            },
        },
    ]


def parse_seed_list(raw: str) -> List[int]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("seed list is empty.")
    return [int(x) for x in values]


def parse_variant_list(raw: str) -> List[str]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("variants is empty.")
    return values


def parse_label_list(raw: str) -> List[str]:
    values = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("split label list is empty.")
    return values


def normalize_split_label(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return ""
    return text


def build_base_config(args: argparse.Namespace) -> model_mod.TrainingConfig:
    cfg = model_mod.TrainingConfig()
    cfg.data_csv_path = args.data_csv_path
    cfg.image_base_dir = args.image_base_dir
    cfg.clip_model_name = args.clip_model_name
    cfg.output_dir = args.output_root
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.w_q = args.w_q
    cfg.w_c = args.w_c
    cfg.test_size = args.test_size
    cfg.val_size_within_train = args.val_size_within_train
    cfg.disable_validation_split = args.no_validation_split
    cfg.use_group_dro = False
    cfg.group_dro_lambda = args.group_dro_lambda
    cfg.bootstrap_iters = args.bootstrap_iters
    cfg.bootstrap_seed = args.bootstrap_seed
    cfg.selection_metric = args.selection_metric
    cfg.group_column = args.group_column
    cfg.device = args.device
    cfg.lora_r = args.lora_r
    cfg.lora_alpha = args.lora_alpha
    cfg.lora_dropout = args.lora_dropout
    cfg.lora_target_modules = args.lora_target_modules
    cfg.text_source = args.text_source
    cfg.use_focus_local_branch = not args.no_focus_local_branch
    cfg.focus_local_scale = args.focus_local_scale
    cfg.focus_local_text_source = args.focus_local_text_source
    cfg.focus_local_fallback_to_global = not args.no_focus_local_fallback_to_global
    cfg.refit_on_trainval = not args.no_refit_trainval
    cfg.group_metric_min_size = args.group_metric_min_size
    cfg.save_val_preds_csv = "preds.csv"
    cfg.save_val_metrics_json = "metrics.json"
    cfg.save_train_log_json = "train_log.json"
    cfg.save_config_json = "config.json"
    cfg.save_checkpoint_name = "checkpoint.pt"
    cfg.save_token_importance_jsonl = ""
    cfg.funnel_cache_jsonl = args.funnel_cache_jsonl or ""
    return cfg


def build_split_file_path(output_root: str, split_seed: int) -> str:
    return os.path.join(os.path.abspath(output_root), "splits", f"split_seed_{split_seed}.json")


def build_existing_split_payload(
        data_csv_path: str,
        split_column: str,
        group_column: str,
        val_size_within_train: float,
        split_seed: int,
        disable_validation_split: bool,
        train_labels: Sequence[str],
        val_labels: Sequence[str],
        test_labels: Sequence[str]) -> Tuple[Dict[str, Any], bool]:
    df = pd.read_csv(data_csv_path)
    df = model_mod.ensure_split_row_ids(df)
    if split_column not in df.columns:
        raise ValueError(f"split_column='{split_column}' not found in {data_csv_path}.")

    normalized = df[split_column].map(normalize_split_label)
    allowed = set(train_labels) | set(val_labels) | set(test_labels)
    unknown = sorted({x for x in normalized.unique().tolist() if x and x not in allowed})
    if unknown:
        raise ValueError(
            f"Unexpected labels in split column '{split_column}': {unknown}. "
            f"Allowed labels are train={list(train_labels)}, val={list(val_labels)}, test={list(test_labels)}."
        )

    train_pool_labels = set(train_labels) | (set(val_labels) if disable_validation_split else set())
    train_pool_df = df[normalized.isin(train_pool_labels)].copy()
    explicit_val_df = df[normalized.isin(set(val_labels))].copy() if not disable_validation_split else df.iloc[0:0].copy()
    test_df = df[normalized.isin(set(test_labels))].copy()

    if train_pool_df.empty:
        raise ValueError(f"No rows matched train labels {list(train_labels)} in column '{split_column}'.")
    if test_df.empty:
        raise ValueError(f"No rows matched test labels {list(test_labels)} in column '{split_column}'.")

    if disable_validation_split:
        train_ids = sorted(int(x) for x in train_pool_df["_split_row_id"].tolist())
        test_ids = sorted(int(x) for x in test_df["_split_row_id"].tolist())
        payload = {
            "version": 3,
            "split_mode": "existing_split_column_train_test_only",
            "split_seed": int(split_seed),
            "split_column": split_column,
            "disable_validation_split": True,
            "train_labels": list(train_labels),
            "val_labels": list(val_labels),
            "test_labels": list(test_labels),
            "num_rows": int(len(df)),
            "train_ids": train_ids,
            "val_ids": [],
            "test_ids": test_ids,
            "effective_train_fraction": float(len(train_ids) / max(len(df), 1)),
            "effective_val_fraction": 0.0,
            "effective_test_fraction": float(len(test_ids) / max(len(df), 1)),
            "outer_split_strategy": f"existing_column:{split_column}",
            "inner_split_strategy": "disabled_no_validation",
        }
        return payload, True

    if not explicit_val_df.empty:
        train_ids = sorted(int(x) for x in train_pool_df["_split_row_id"].tolist())
        val_ids = sorted(int(x) for x in explicit_val_df["_split_row_id"].tolist())
        test_ids = sorted(int(x) for x in test_df["_split_row_id"].tolist())
        payload = {
            "version": 3,
            "split_mode": "existing_split_column_three_way",
            "split_seed": int(split_seed),
            "split_column": split_column,
            "train_labels": list(train_labels),
            "val_labels": list(val_labels),
            "test_labels": list(test_labels),
            "num_rows": int(len(df)),
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": test_ids,
            "effective_train_fraction": float(len(train_ids) / max(len(df), 1)),
            "effective_val_fraction": float(len(val_ids) / max(len(df), 1)),
            "effective_test_fraction": float(len(test_ids) / max(len(df), 1)),
            "outer_split_strategy": f"existing_column:{split_column}",
            "inner_split_strategy": "provided_val_labels",
        }
        return payload, True

    annotated_train_pool, split_metadata = model_mod.annotate_split_context(train_pool_df, group_column=group_column)
    inner_split_seed = int(split_seed + 1009)
    train_ids, val_ids, inner_strategy = model_mod._split_ids_with_candidates(
        annotated_train_pool,
        heldout_size=val_size_within_train,
        random_state=inner_split_seed,
    )
    test_ids = sorted(int(x) for x in test_df["_split_row_id"].tolist())
    payload = {
        "version": 3,
        "split_mode": "existing_split_column_train_test",
        "split_seed": int(split_seed),
        "inner_split_seed": inner_split_seed,
        "split_column": split_column,
        "train_labels": list(train_labels),
        "val_labels": list(val_labels),
        "test_labels": list(test_labels),
        "num_rows": int(len(df)),
        "train_ids": sorted(int(x) for x in train_ids),
        "val_ids": sorted(int(x) for x in val_ids),
        "test_ids": test_ids,
        "effective_train_fraction": float(len(train_ids) / max(len(df), 1)),
        "effective_val_fraction": float(len(val_ids) / max(len(df), 1)),
        "effective_test_fraction": float(len(test_ids) / max(len(df), 1)),
        "outer_split_strategy": f"existing_column:{split_column}",
        "inner_split_strategy": inner_strategy,
    }
    payload.update({k: model_mod._json_ready(v) for k, v in split_metadata.items()})
    return payload, False


def materialize_existing_split_files(
        output_root: str,
        data_csv_path: str,
        split_column: str,
        group_column: str,
        val_size_within_train: float,
        requested_split_seeds: Sequence[int],
        disable_validation_split: bool,
        train_labels: Sequence[str],
        val_labels: Sequence[str],
        test_labels: Sequence[str]) -> Tuple[List[int], Dict[str, str], str]:
    initial_payload, has_explicit_val = build_existing_split_payload(
        data_csv_path=data_csv_path,
        split_column=split_column,
        group_column=group_column,
        val_size_within_train=val_size_within_train,
        split_seed=int(requested_split_seeds[0]),
        disable_validation_split=disable_validation_split,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
    )

    if disable_validation_split:
        effective_split_seeds = [int(requested_split_seeds[0])]
        split_description = (
            f"CSV 固定 train/test，来自列 '{split_column}' "
            f"(train={list(train_labels)}, test={list(test_labels)})；不使用 validation，最终 epoch 用于测试"
        )
    elif has_explicit_val:
        effective_split_seeds = [int(requested_split_seeds[0])]
        split_description = (
            f"CSV 固定三路划分，来自列 '{split_column}' "
            f"(train={list(train_labels)}, val={list(val_labels)}, test={list(test_labels)})"
        )
    else:
        effective_split_seeds = list(requested_split_seeds)
        split_description = (
            f"CSV 固定 train/test，来自列 '{split_column}' "
            f"(train={list(train_labels)}, test={list(test_labels)}); "
            f"val 在 train 池内按 split_seed 再划分"
        )

    split_files: Dict[str, str] = {}
    for split_seed in effective_split_seeds:
        payload, _ = build_existing_split_payload(
            data_csv_path=data_csv_path,
            split_column=split_column,
            group_column=group_column,
            val_size_within_train=val_size_within_train,
            split_seed=split_seed,
            disable_validation_split=disable_validation_split,
            train_labels=train_labels,
            val_labels=val_labels,
            test_labels=test_labels,
        )
        split_path = build_split_file_path(output_root, split_seed)
        split_parent = os.path.dirname(split_path)
        if split_parent:
            os.makedirs(split_parent, exist_ok=True)
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        split_files[str(split_seed)] = split_path
    return effective_split_seeds, split_files, split_description


def build_run_artifact_paths(
        output_root: str,
        split_seed: Any,
        variant_name: Optional[str] = None,
        seed: Optional[int] = None,
        base_cfg: Optional[model_mod.TrainingConfig] = None) -> Dict[str, str]:
    if base_cfg is None:
        base_cfg = seed  # type: ignore[assignment]
        seed = variant_name  # type: ignore[assignment]
        variant_name = split_seed  # type: ignore[assignment]
        split_seed = 42
    assert variant_name is not None
    assert seed is not None
    assert base_cfg is not None
    run_dir = os.path.join(os.path.abspath(output_root), f"split_{split_seed}", variant_name, f"seed_{seed}")
    return {
        "run_dir": run_dir,
        "preds_path": os.path.join(run_dir, os.path.basename(base_cfg.save_val_preds_csv)),
        "metrics_path": os.path.join(run_dir, os.path.basename(base_cfg.save_val_metrics_json)),
        "train_log_path": os.path.join(run_dir, os.path.basename(base_cfg.save_train_log_json)),
        "config_path": os.path.join(run_dir, os.path.basename(base_cfg.save_config_json)),
        "checkpoint_path": os.path.join(run_dir, os.path.basename(base_cfg.save_checkpoint_name)),
    }


def is_completed_run(run_paths: Dict[str, str]) -> bool:
    required_keys = ["preds_path", "metrics_path", "train_log_path", "config_path", "checkpoint_path"]
    for key in required_keys:
        path = run_paths[key]
        if not os.path.exists(path):
            return False
        if key != "checkpoint_path" and os.path.getsize(path) <= 0:
            return False
    return True


def load_completed_run_result(run_paths: Dict[str, str]) -> Dict[str, Any]:
    with open(run_paths["metrics_path"], "r", encoding="utf-8") as f:
        metrics = json.load(f)
    split_payload = None
    if os.path.exists(run_paths["train_log_path"]):
        with open(run_paths["train_log_path"], "r", encoding="utf-8") as f:
            train_log = json.load(f)
        split_payload = train_log.get("split")
    return {
        "run_dir": run_paths["run_dir"],
        "checkpoint_path": run_paths["checkpoint_path"],
        "preds_path": run_paths["preds_path"],
        "metrics_path": run_paths["metrics_path"],
        "train_log_path": run_paths["train_log_path"],
        "config_path": run_paths["config_path"],
        "metrics": metrics,
        "split": split_payload,
    }


def load_run_config_metadata(result: Dict[str, Any]) -> Dict[str, Any]:
    config_path = result.get("config_path", "")
    if not config_path or not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {key: cfg.get(key) for key in FOCUS_METADATA_KEYS}


def saved_run_matches_config(run_paths: Dict[str, str], cfg: model_mod.TrainingConfig) -> bool:
    if not os.path.exists(run_paths["config_path"]):
        return False
    with open(run_paths["config_path"], "r", encoding="utf-8") as f:
        saved_cfg = json.load(f)
    for key in CONFIG_MATCH_KEYS:
        if saved_cfg.get(key) != getattr(cfg, key, None):
            return False
    return True


def summarize_run(variant_name: str, split_seed: int, seed: int, result: Dict[str, Any]) -> Dict[str, Any]:
    metrics = result["metrics"]
    focus_meta = load_run_config_metadata(result)
    return {
        "variant": variant_name,
        "split_seed": split_seed,
        "seed": seed,
        "run_dir": result["run_dir"],
        "preds_path": result["preds_path"],
        "metrics_path": result["metrics_path"],
        "selection_score": metrics["selection_score"],
        "quality_srocc": metrics["quality"]["srocc"]["value"],
        "quality_plcc": metrics["quality"]["plcc"]["value"],
        "quality_rmse": metrics["quality"]["rmse"]["value"],
        "quality_mae": metrics["quality"]["mae"]["value"],
        "consistency_srocc": metrics["consistency"]["srocc"]["value"],
        "consistency_plcc": metrics["consistency"]["plcc"]["value"],
        "consistency_rmse": metrics["consistency"]["rmse"]["value"],
        "consistency_mae": metrics["consistency"]["mae"]["value"],
        "worst_group_c_srocc": metrics["consistency"]["group_summary"]["worst_group_c_srocc"],
        "worst_group_c_rmse": metrics["consistency"]["group_summary"]["worst_group_c_rmse"],
        "mean_group_c_srocc": metrics["consistency"]["group_summary"]["mean_group_c_srocc"],
        "trainable_params": metrics.get("trainable_params", 0),
        "total_params": metrics.get("total_params", 0),
        "runtime_sec": metrics.get("runtime_sec", 0.0),
        "best_epoch": metrics.get("best_epoch", -1),
        "evaluation_split": metrics.get("evaluation_split", "test"),
        "use_focus_local_branch": focus_meta.get("use_focus_local_branch"),
        "focus_local_scale": focus_meta.get("focus_local_scale"),
        "focus_local_text_source": focus_meta.get("focus_local_text_source"),
        "focus_local_fallback_to_global": focus_meta.get("focus_local_fallback_to_global"),
    }


def aggregate_metrics_rows(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    numeric_cols = [
        "selection_score",
        "quality_srocc",
        "quality_plcc",
        "quality_rmse",
        "quality_mae",
        "consistency_srocc",
        "consistency_plcc",
        "consistency_rmse",
        "consistency_mae",
        "worst_group_c_srocc",
        "worst_group_c_rmse",
        "mean_group_c_srocc",
        "trainable_params",
        "total_params",
        "runtime_sec",
    ]
    agg = df.groupby("variant", sort=False)[numeric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["variant"] + [f"{name}_{stat}" for name, stat in agg.columns.tolist()[1:]]
    metadata_cols = ["variant"] + [name for name in FOCUS_METADATA_KEYS if name in df.columns]
    if len(metadata_cols) > 1:
        metadata_df = df[metadata_cols].drop_duplicates(subset=["variant"], keep="first")
        agg = agg.merge(metadata_df, on="variant", how="left")
    variant_order = {spec["name"]: idx for idx, spec in enumerate(build_ablation_variants())}
    agg["_variant_order"] = agg["variant"].map(lambda name: variant_order.get(name, len(variant_order)))
    agg = agg.sort_values(["_variant_order", "variant"]).drop(columns=["_variant_order"]).reset_index(drop=True)
    return agg


def load_pred_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "_split_row_id" in df.columns:
        return df.sort_values("_split_row_id").reset_index(drop=True)
    return df.sort_values("name").reset_index(drop=True)


def choose_strongest_baseline(
        aggregate_df: pd.DataFrame,
        candidate_variants: Optional[Sequence[str]] = None) -> Optional[str]:
    if aggregate_df.empty:
        return None
    candidates = list(candidate_variants or BASELINE_CANDIDATE_VARIANTS)
    source = aggregate_df.set_index("variant")
    present = [name for name in candidates if name in source.index]
    if not present:
        return None
    ranked = sorted(
        present,
        key=lambda name: (
            float(source.loc[name, "consistency_srocc_mean"]),
            float(source.loc[name, "worst_group_c_srocc_mean"]),
            float(source.loc[name, "quality_srocc_mean"]),
        ),
        reverse=True,
    )
    return ranked[0]


def resolve_comparison_specs(aggregate_df: pd.DataFrame) -> List[Dict[str, Any]]:
    available = set(aggregate_df["variant"].tolist()) if not aggregate_df.empty else set()
    strongest_baseline = choose_strongest_baseline(aggregate_df)
    resolved: List[Dict[str, Any]] = []
    for spec in COMPARISON_SPECS:
        if spec["mode"] == "auto_strongest_baseline":
            if strongest_baseline is None or spec["candidate_variant"] not in available:
                continue
            resolved.append(
                {
                    "name": spec["name"],
                    "label": spec["label"],
                    "reference_variant": strongest_baseline,
                    "candidate_variant": spec["candidate_variant"],
                }
            )
            continue
        if spec["reference_variant"] in available and spec["candidate_variant"] in available:
            resolved.append(
                {
                    "name": spec["name"],
                    "label": spec["label"],
                    "reference_variant": spec["reference_variant"],
                    "candidate_variant": spec["candidate_variant"],
                }
            )
    return resolved


def _load_run_prediction_arrays(run_row: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    df = load_pred_df(run_row["preds_path"])
    return df["target_c"].to_numpy(dtype=np.float64), df["pred_c"].to_numpy(dtype=np.float64)


def _delta_row(
        comparison_name: str,
        comparison_label: str,
        reference_variant: str,
        candidate_variant: str,
        split_seed: int,
        seed: int,
        y: np.ndarray,
        ref_pred: np.ndarray,
        cmp_pred: np.ndarray,
        bootstrap_iters: int,
        bootstrap_seed: int) -> Dict[str, Any]:
    return {
        "comparison": comparison_name,
        "comparison_label": comparison_label,
        "reference_variant": reference_variant,
        "candidate_variant": candidate_variant,
        "split_seed": split_seed,
        "seed": seed,
        "target": "consistency",
        "delta_srocc": model_mod.paired_bootstrap_delta(
            y, ref_pred, cmp_pred, kind="spearman", n_boot=bootstrap_iters, seed=bootstrap_seed
        ),
        "delta_plcc": model_mod.paired_bootstrap_delta(
            y, ref_pred, cmp_pred, kind="pearson", n_boot=bootstrap_iters, seed=bootstrap_seed
        ),
        "delta_mae": model_mod.paired_bootstrap_delta(
            y, ref_pred, cmp_pred, kind="mae", n_boot=bootstrap_iters, seed=bootstrap_seed
        ),
        "delta_rmse": model_mod.paired_bootstrap_delta(
            y, ref_pred, cmp_pred, kind="rmse", n_boot=bootstrap_iters, seed=bootstrap_seed
        ),
    }


def compute_delta_rows(
        run_rows: Sequence[Dict[str, Any]],
        comparison_specs: Optional[Sequence[Dict[str, Any]]] = None,
        bootstrap_iters: int = 5000,
        bootstrap_seed: int = 42) -> List[Dict[str, Any]]:
    normalized_rows = []
    for row in run_rows:
        copied = dict(row)
        copied.setdefault("split_seed", 42)
        normalized_rows.append(copied)
    if comparison_specs is None:
        comparison_specs = resolve_comparison_specs(aggregate_metrics_rows(normalized_rows))
    by_variant_split_seed = {(row["variant"], row["split_seed"], row["seed"]): row for row in normalized_rows}
    delta_rows: List[Dict[str, Any]] = []
    split_seeds = sorted({row["split_seed"] for row in normalized_rows})
    train_seeds = sorted({row["seed"] for row in normalized_rows})

    for spec in comparison_specs:
        for split_seed in split_seeds:
            for seed in train_seeds:
                ref_row = by_variant_split_seed.get((spec["reference_variant"], split_seed, seed))
                cmp_row = by_variant_split_seed.get((spec["candidate_variant"], split_seed, seed))
                if ref_row is None or cmp_row is None:
                    continue
                y, ref_pred = _load_run_prediction_arrays(ref_row)
                _, cmp_pred = _load_run_prediction_arrays(cmp_row)
                if len(ref_pred) != len(cmp_pred):
                    raise ValueError(
                        f"Mismatched prediction length for {spec['candidate_variant']} vs {spec['reference_variant']} "
                        f"(split {split_seed}, seed {seed})."
                    )
                delta_rows.append(
                    _delta_row(
                        comparison_name=spec["name"],
                        comparison_label=spec["label"],
                        reference_variant=spec["reference_variant"],
                        candidate_variant=spec["candidate_variant"],
                        split_seed=split_seed,
                        seed=seed,
                        y=y,
                        ref_pred=ref_pred,
                        cmp_pred=cmp_pred,
                        bootstrap_iters=bootstrap_iters,
                        bootstrap_seed=bootstrap_seed,
                    )
                )
    return delta_rows


def paired_bootstrap_mean_delta(
        arrays: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        kind: str,
        n_boot: int,
        seed: int) -> Dict[str, Any]:
    if not arrays:
        return {
            "mean_delta": 0.0,
            "ci95": [0.0, 0.0],
            "crosses_zero": True,
            "metric": kind,
            "num_pairs": 0,
        }

    point_deltas = [
        model_mod._metric_value(y, cand_pred, kind) - model_mod._metric_value(y, ref_pred, kind)
        for y, ref_pred, cand_pred in arrays
    ]
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        batch_deltas = []
        for y, ref_pred, cand_pred in arrays:
            idx = rng.integers(0, len(y), size=len(y))
            batch_deltas.append(
                model_mod._metric_value(y[idx], cand_pred[idx], kind)
                - model_mod._metric_value(y[idx], ref_pred[idx], kind)
            )
        boot.append(float(np.mean(batch_deltas)))
    lo = float(np.quantile(boot, 0.025))
    hi = float(np.quantile(boot, 0.975))
    return {
        "mean_delta": float(np.mean(point_deltas)),
        "ci95": [lo, hi],
        "crosses_zero": bool(lo <= 0.0 <= hi),
        "metric": kind,
        "num_pairs": int(len(arrays)),
    }


def build_delta_summary(
        run_rows: Sequence[Dict[str, Any]],
        comparison_specs: Sequence[Dict[str, Any]],
        bootstrap_iters: int,
        bootstrap_seed: int) -> pd.DataFrame:
    by_variant_split_seed = {(row["variant"], row["split_seed"], row["seed"]): row for row in run_rows}
    split_seeds = sorted({row["split_seed"] for row in run_rows})
    train_seeds = sorted({row["seed"] for row in run_rows})
    rows: List[Dict[str, Any]] = []

    for spec in comparison_specs:
        matched_arrays: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for split_seed in split_seeds:
            for seed in train_seeds:
                ref_row = by_variant_split_seed.get((spec["reference_variant"], split_seed, seed))
                cmp_row = by_variant_split_seed.get((spec["candidate_variant"], split_seed, seed))
                if ref_row is None or cmp_row is None:
                    continue
                y, ref_pred = _load_run_prediction_arrays(ref_row)
                _, cmp_pred = _load_run_prediction_arrays(cmp_row)
                matched_arrays.append((y, ref_pred, cmp_pred))
        if not matched_arrays:
            continue
        for label, kind in [
            ("C-SROCC", "spearman"),
            ("C-PLCC", "pearson"),
            ("C-MAE", "mae"),
            ("C-RMSE", "rmse"),
        ]:
            payload = paired_bootstrap_mean_delta(
                matched_arrays,
                kind=kind,
                n_boot=bootstrap_iters,
                seed=bootstrap_seed,
            )
            rows.append(
                {
                    "comparison": spec["name"],
                    "comparison_label": spec["label"],
                    "reference_variant": spec["reference_variant"],
                    "candidate_variant": spec["candidate_variant"],
                    "metric": label,
                    "mean_delta": payload["mean_delta"],
                    "ci95_lo": payload["ci95"][0],
                    "ci95_hi": payload["ci95"][1],
                    "crosses_zero": payload["crosses_zero"],
                    "num_pairs": payload["num_pairs"],
                }
            )
    return pd.DataFrame(rows)


def flatten_group_metrics(run_rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for row in run_rows:
        with open(row["metrics_path"], "r", encoding="utf-8") as f:
            metrics = json.load(f)
        for group_metric in metrics.get("group_metrics", []):
            rows.append(
                {
                    "variant": row["variant"],
                    "split_seed": row["split_seed"],
                    "seed": row["seed"],
                    **group_metric,
                }
            )
    return pd.DataFrame(rows)


def _fmt_mean_std(mean_value: float, std_value: float) -> str:
    std = 0.0 if pd.isna(std_value) else float(std_value)
    return f"{float(mean_value):.4f} +/- {std:.4f}"


def _fmt_params_millions(param_count: float) -> str:
    return f"{float(param_count) / 1e6:.2f}"


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data_"
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(df.iloc[i][col]) for col in columns) + " |" for i in range(len(df))]
    return "\n".join([header, separator] + rows)


def _table_from_aggregate(agg_df: pd.DataFrame, variants: Iterable[str]) -> pd.DataFrame:
    if agg_df.empty:
        return pd.DataFrame()
    source = agg_df.set_index("variant")
    rows: List[Dict[str, str]] = []
    for variant in variants:
        if variant not in source.index:
            continue
        row = source.loc[variant]
        rows.append(
            {
                "Variant": DISPLAY_NAME_BY_VARIANT.get(variant, variant),
                "Focus-Local": "On" if bool(row.get("use_focus_local_branch", False)) else "Off",
                "Q-SROCC": _fmt_mean_std(row["quality_srocc_mean"], row["quality_srocc_std"]),
                "Q-PLCC": _fmt_mean_std(row["quality_plcc_mean"], row["quality_plcc_std"]),
                "C-SROCC": _fmt_mean_std(row["consistency_srocc_mean"], row["consistency_srocc_std"]),
                "C-PLCC": _fmt_mean_std(row["consistency_plcc_mean"], row["consistency_plcc_std"]),
                "C-MAE": _fmt_mean_std(row["consistency_mae_mean"], row["consistency_mae_std"]),
                "C-RMSE": _fmt_mean_std(row["consistency_rmse_mean"], row["consistency_rmse_std"]),
                "Worst-Group C-SROCC": _fmt_mean_std(row["worst_group_c_srocc_mean"], row["worst_group_c_srocc_std"]),
                "Trainable Params (M)": _fmt_params_millions(row["trainable_params_mean"]),
                "Runtime (s)": _fmt_mean_std(row["runtime_sec_mean"], row["runtime_sec_std"]),
            }
        )
    return pd.DataFrame(rows)


def build_variant_metadata_rows(agg_df: pd.DataFrame) -> pd.DataFrame:
    if agg_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for _, row in agg_df.iterrows():
        rows.append(
            {
                "Variant": DISPLAY_NAME_BY_VARIANT.get(row["variant"], row["variant"]),
                "Focus-Local": "On" if bool(row.get("use_focus_local_branch", False)) else "Off",
                "Focus Scale": row.get("focus_local_scale", ""),
                "Focus Text Source": row.get("focus_local_text_source", ""),
                "Focus Fallback": "Yes" if bool(row.get("focus_local_fallback_to_global", False)) else "No",
            }
        )
    return pd.DataFrame(rows)


def build_main_report_rows(agg_df: pd.DataFrame) -> pd.DataFrame:
    variant_names = [spec["name"] for spec in build_ablation_variants()]
    return _table_from_aggregate(agg_df, variant_names)


def build_delta_report_rows(delta_summary_df: pd.DataFrame) -> pd.DataFrame:
    if delta_summary_df.empty:
        return pd.DataFrame()
    report_df = delta_summary_df.copy()
    report_df["Mean Delta"] = report_df["mean_delta"].map(lambda x: f"{float(x):+.4f}")
    report_df["95% CI"] = report_df.apply(
        lambda row: f"[{float(row['ci95_lo']):+.4f}, {float(row['ci95_hi']):+.4f}]",
        axis=1,
    )
    report_df["Crosses 0"] = report_df["crosses_zero"].map(lambda x: "Yes" if x else "No")
    return report_df[["comparison_label", "metric", "Mean Delta", "95% CI", "Crosses 0", "num_pairs"]].rename(
        columns={
            "comparison_label": "Comparison",
            "metric": "Metric",
            "num_pairs": "Runs",
        }
    )


def write_report(
        output_root: str,
        train_seeds: Sequence[int],
        split_seeds: Sequence[int],
        aggregate_df: pd.DataFrame,
        delta_summary_df: pd.DataFrame,
        strongest_baseline: Optional[str],
        split_description: str,
        selection_description: str) -> None:
    main_table = build_main_report_rows(aggregate_df)
    metadata_table = build_variant_metadata_rows(aggregate_df)
    delta_table = build_delta_report_rows(delta_summary_df)

    if not main_table.empty:
        main_table.to_csv(os.path.join(output_root, "report_main_table.csv"), index=False, encoding="utf-8")
    if not metadata_table.empty:
        metadata_table.to_csv(os.path.join(output_root, "report_variant_metadata.csv"), index=False, encoding="utf-8")
    if not delta_table.empty:
        delta_table.to_csv(os.path.join(output_root, "report_delta_summary.csv"), index=False, encoding="utf-8")

    strongest_baseline_text = DISPLAY_NAME_BY_VARIANT.get(strongest_baseline, strongest_baseline) if strongest_baseline else "N/A"
    report_path = os.path.join(output_root, "report.md")
    lines = [
        "# ONSRC Multi-Split Ablation Report",
        "",
        f"- Split seeds: `{','.join(str(x) for x in split_seeds)}`",
        f"- Training seeds: `{','.join(str(x) for x in train_seeds)}`",
        f"- Split protocol: {split_description}",
        f"- Checkpoint selection: {selection_description}",
        "- Main table metrics are reported on the held-out test split as mean +/- std over all split/seed runs",
        "- Focus-local branch: explicit runner-controlled config; local CLIP score is blended into consistency prediction before residual calibration",
        f"- Strongest fair rerun baseline: `{strongest_baseline_text}`",
        "",
        "## Variant Metadata",
        "",
        _markdown_table(metadata_table),
        "",
        "## Main Table",
        "",
        _markdown_table(main_table),
        "",
        "## Paired Bootstrap Summary",
        "",
        "Each row reports candidate-minus-reference mean delta over matched split/seed test runs, 95% paired bootstrap CI, and whether the CI crosses 0.",
        "",
        _markdown_table(delta_table),
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser("Run ONSRC ablations with multi-split evaluation")
    parser.add_argument("--data_csv_path", required=True)
    parser.add_argument("--image_base_dir", required=True)
    parser.add_argument("--clip_model_name", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--w_q", type=float, default=0.3)
    parser.add_argument("--w_c", type=float, default=0.7)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size_within_train", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if model_mod.torch.cuda.is_available() else "cpu")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,out_proj")
    parser.add_argument("--group_dro_lambda", type=float, default=0.0)
    parser.add_argument("--group_dro_lambda_ablation", type=float, default=0.0)
    parser.add_argument("--bootstrap_iters", type=int, default=5000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument("--split_seeds", default="42,52,62")
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--use_existing_split_column", action="store_true")
    parser.add_argument("--split_column", default="split")
    parser.add_argument("--split_train_labels", default="train")
    parser.add_argument("--split_val_labels", default="val,valid,validation,dev")
    parser.add_argument("--split_test_labels", default="test")
    parser.add_argument("--no_validation_split", action="store_true")
    parser.add_argument("--train_seeds", default="11,22,33")
    parser.add_argument(
        "--selection_metric",
        default="weighted_qc_srocc",
        choices=["avg_srocc", "weighted_qc_srocc", "consistency_srocc", "quality_srocc"],
    )
    parser.add_argument("--group_column", default="auto")
    parser.add_argument("--group_metric_min_size", type=int, default=3)
    parser.add_argument("--funnel_cache_jsonl", default="")
    parser.add_argument("--text_source", default="raw_prompt", choices=getattr(model_mod, "TEXT_SOURCE_CHOICES", ("raw_prompt",)))
    parser.add_argument("--no_focus_local_branch", action="store_true")
    parser.add_argument("--focus_local_scale", type=float, default=1.0)
    parser.add_argument(
        "--focus_local_text_source",
        default="funnel_selected_prompt",
        choices=getattr(model_mod, "TEXT_SOURCE_CHOICES", ("raw_prompt",)),
    )
    parser.add_argument("--no_focus_local_fallback_to_global", action="store_true")
    parser.add_argument("--no_refit_trainval", action="store_true")
    parser.add_argument("--variants", default="")
    parser.add_argument("--force_rerun", action="store_true")
    args = parser.parse_args()

    if args.lora_alpha is None:
        args.lora_alpha = args.lora_r * 2

    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(os.path.join(output_root, "splits"), exist_ok=True)

    requested_split_seeds = [args.split_seed] if args.split_seed is not None else parse_seed_list(args.split_seeds)
    train_seeds = parse_seed_list(args.train_seeds)
    base_cfg = build_base_config(args)

    variant_specs = build_ablation_variants()
    if args.variants.strip():
        requested_variants = parse_variant_list(args.variants)
        by_name = {spec["name"]: spec for spec in variant_specs}
        missing = [name for name in requested_variants if name not in by_name]
        if missing:
            raise ValueError(f"Unknown variants requested: {missing}")
        variant_specs = [by_name[name] for name in requested_variants]

    if args.use_existing_split_column:
        train_labels = parse_label_list(args.split_train_labels)
        val_labels = parse_label_list(args.split_val_labels)
        test_labels = parse_label_list(args.split_test_labels)
        split_seeds, split_files, split_description = materialize_existing_split_files(
            output_root=output_root,
            data_csv_path=args.data_csv_path,
            split_column=args.split_column,
            group_column=args.group_column,
            val_size_within_train=args.val_size_within_train,
            requested_split_seeds=requested_split_seeds,
            disable_validation_split=args.no_validation_split,
            train_labels=train_labels,
            val_labels=val_labels,
            test_labels=test_labels,
        )
        if len(split_seeds) == 1 and len(requested_split_seeds) > 1:
            print("[Info] CSV split 为固定划分，split_seeds 被折叠为单一划分。")
    else:
        split_seeds = requested_split_seeds
        split_files = {str(seed): build_split_file_path(output_root, seed) for seed in split_seeds}
        if args.no_validation_split:
            split_description = "random train-test split only; validation disabled and final epoch is used for testing"
        else:
            split_description = "outer 80/20 train-test, inner 90/10 train-validation inside the training pool"

    if args.no_validation_split:
        selection_description = "validation disabled; final epoch checkpoint is evaluated on the test split"
    else:
        selection_description = "validation `0.7 * C-SROCC + 0.3 * Q-SROCC`"

    split_manifest = {
        "split_mode": "existing_split_column" if args.use_existing_split_column else "random_nested",
        "split_description": split_description,
        "selection_description": selection_description,
        "split_seeds": split_seeds,
        "train_seeds": train_seeds,
        "split_files": split_files,
    }
    if args.use_existing_split_column:
        split_manifest.update(
            {
                "split_column": args.split_column,
                "split_train_labels": parse_label_list(args.split_train_labels),
                "split_val_labels": parse_label_list(args.split_val_labels),
                "split_test_labels": parse_label_list(args.split_test_labels),
                "no_validation_split": args.no_validation_split,
            }
        )
    else:
        split_manifest["no_validation_split"] = args.no_validation_split
    with open(os.path.join(output_root, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, ensure_ascii=False, indent=2)

    run_rows: List[Dict[str, Any]] = []
    for split_seed in split_seeds:
        split_file = split_files[str(split_seed)]
        for spec in variant_specs:
            for seed in train_seeds:
                cfg = copy.deepcopy(base_cfg)
                cfg.seed = seed
                cfg.split_seed = split_seed
                cfg.split_file = split_file
                cfg.run_name = os.path.join(f"split_{split_seed}", spec["name"], f"seed_{seed}")
                for key, value in spec["overrides"].items():
                    setattr(cfg, key, value)

                run_paths = build_run_artifact_paths(output_root, split_seed, spec["name"], seed, base_cfg)
                run_label = f"split_{split_seed}/{spec['name']}/seed_{seed}"

                if (
                        not args.force_rerun
                        and is_completed_run(run_paths)
                        and saved_run_matches_config(run_paths, cfg)):
                    print(f"[Skip] Reusing completed run: {run_label}")
                    result = load_completed_run_result(run_paths)
                else:
                    if (
                            not args.force_rerun
                            and os.path.isdir(run_paths["run_dir"])
                            and any(Path(run_paths["run_dir"]).iterdir())):
                        if is_completed_run(run_paths):
                            print(f"[Rerun] Found completed run with stale config, rerunning: {run_label}")
                        else:
                            print(f"[Resume] Found incomplete run, rerunning: {run_label}")
                    result = model_mod.run_training(cfg)

                run_rows.append(summarize_run(spec["name"], split_seed, seed, result))

    metrics_by_seed_df = pd.DataFrame(run_rows)
    metrics_by_seed_df.to_csv(os.path.join(output_root, "metrics_by_seed.csv"), index=False, encoding="utf-8")

    aggregate_df = aggregate_metrics_rows(run_rows)
    aggregate_df.to_csv(os.path.join(output_root, "aggregate.csv"), index=False, encoding="utf-8")

    comparison_specs = resolve_comparison_specs(aggregate_df)
    delta_rows = compute_delta_rows(
        run_rows,
        comparison_specs=comparison_specs,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_seed=args.bootstrap_seed,
    )
    delta_df = pd.DataFrame(delta_rows)
    if not delta_df.empty:
        flat_delta_df = delta_df.copy()
        for metric_name in ["delta_srocc", "delta_plcc", "delta_mae", "delta_rmse"]:
            flat_delta_df[f"{metric_name}_value"] = flat_delta_df[metric_name].map(lambda x: x["delta"])
            flat_delta_df[f"{metric_name}_ci95_lo"] = flat_delta_df[metric_name].map(lambda x: x["ci95"][0])
            flat_delta_df[f"{metric_name}_ci95_hi"] = flat_delta_df[metric_name].map(lambda x: x["ci95"][1])
        flat_delta_df = flat_delta_df.drop(columns=["delta_srocc", "delta_plcc", "delta_mae", "delta_rmse"])
    else:
        flat_delta_df = pd.DataFrame(
            columns=[
                "comparison",
                "comparison_label",
                "reference_variant",
                "candidate_variant",
                "split_seed",
                "seed",
                "target",
                "delta_srocc_value",
                "delta_srocc_ci95_lo",
                "delta_srocc_ci95_hi",
                "delta_plcc_value",
                "delta_plcc_ci95_lo",
                "delta_plcc_ci95_hi",
                "delta_mae_value",
                "delta_mae_ci95_lo",
                "delta_mae_ci95_hi",
                "delta_rmse_value",
                "delta_rmse_ci95_lo",
                "delta_rmse_ci95_hi",
            ]
        )
    flat_delta_df.to_csv(os.path.join(output_root, "delta_vs_ref.csv"), index=False, encoding="utf-8")

    delta_summary_df = build_delta_summary(
        run_rows,
        comparison_specs=comparison_specs,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_seed=args.bootstrap_seed,
    )
    delta_summary_df.to_csv(os.path.join(output_root, "delta_summary.csv"), index=False, encoding="utf-8")

    group_metrics_df = flatten_group_metrics(run_rows)
    group_metrics_df.to_csv(os.path.join(output_root, "group_metrics.csv"), index=False, encoding="utf-8")

    strongest_baseline = choose_strongest_baseline(aggregate_df)
    write_report(
        output_root=output_root,
        train_seeds=train_seeds,
        split_seeds=split_seeds,
        aggregate_df=aggregate_df,
        delta_summary_df=delta_summary_df,
        strongest_baseline=strongest_baseline,
        split_description=split_description,
        selection_description=selection_description,
    )


if __name__ == "__main__":
    main()
