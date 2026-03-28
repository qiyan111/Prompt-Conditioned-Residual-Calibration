#!/usr/bin/env python3
"""Run external fair-rerun baselines under the same split protocol."""

import argparse
import inspect
import json
import os
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
IMPORT_ROOTS = [SCRIPT_DIR, os.path.dirname(SCRIPT_DIR)]
for import_root in IMPORT_ROOTS:
    if import_root and import_root not in sys.path:
        sys.path.insert(0, import_root)

import run_model_ablations as ablation_mod
import train as model_mod


DEFAULT_SPLIT_SEEDS = ablation_mod.DEFAULT_SPLIT_SEEDS
DEFAULT_TRAIN_SEEDS = ablation_mod.DEFAULT_TRAIN_SEEDS

VARIANT_SPECS = [
    {
        "name": "IPCE",
        "backend": "ipce",
        "default_env_name": "IPCE",
        "default_repo_name": "IPCE",
        "supported_tasks": {"quality", "alignment"},
    },
    {
        "name": "CLIP-AGIQA",
        "backend": "clip_agiqa",
        "default_env_name": "CLIP-AGIQA",
        "default_repo_name": "CLIP-AGIQA",
        "supported_tasks": {"quality"},
    },
    {
        "name": "MA-AGIQA",
        "backend": "ma_agiqa",
        "default_env_name": "MA-AGIQA",
        "default_repo_name": "MA-AGIQA",
        "supported_tasks": {"quality", "alignment"},
    },
    {
        "name": "M3-AGIQA",
        "backend": "m3_agiqa",
        "default_env_name": "M3-AGIQA",
        "default_repo_name": "M3-AGIQA",
        "supported_tasks": {"quality", "alignment"},
    },
]

DISPLAY_NAME_BY_VARIANT = {spec["name"]: spec["name"] for spec in VARIANT_SPECS}
RUN_CONFIG_MATCH_KEYS = [
    "variant",
    "backend",
    "task",
    "label_column",
    "repo_dir",
    "conda_env_name",
    "data_csv_path",
    "image_base_dir",
    "device",
    "epochs",
    "batch_size",
    "lr",
    "num_workers",
    "test_size",
    "val_size_within_train",
    "no_validation_split",
    "train_loss_stop_threshold",
    "split_seed",
    "seed",
    "split_file",
    "refit_on_trainval",
    "tensor_root",
    "base_config_path",
    "train_pool_json",
    "eval_pool_json",
    "test_pool_json",
    "dataset_name",
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


def label_column_for_task(task: str) -> str:
    if task == "quality":
        return "mos_quality"
    if task == "alignment":
        return "mos_align"
    raise ValueError(f"Unsupported task='{task}'.")


def annotate_split_context_compat(df: pd.DataFrame, group_column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    annotate_fn = getattr(model_mod, "annotate_split_context", None)
    if callable(annotate_fn):
        return annotate_fn(df, group_column=group_column)
    return df, {}


def load_or_create_nested_split_compat(
    df: pd.DataFrame,
    test_size: float,
    val_size_within_train: float,
    split_seed: int,
    split_file: str,
    split_metadata: Optional[Dict[str, Any]],
    disable_validation_split: bool,
) -> Tuple[List[int], List[int], List[int], Dict[str, Any]]:
    split_fn = model_mod.load_or_create_nested_split
    kwargs: Dict[str, Any] = {
        "df": df,
        "test_size": test_size,
        "val_size_within_train": val_size_within_train,
        "split_seed": split_seed,
        "split_file": split_file,
        "disable_validation_split": disable_validation_split,
    }
    if "split_metadata" in inspect.signature(split_fn).parameters:
        kwargs["split_metadata"] = split_metadata
    return split_fn(**kwargs)


def compute_group_metrics_compat(
    pred_df: pd.DataFrame,
    group_metric_min_size: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    compute_fn = model_mod._compute_group_metrics
    compat_df = pred_df.copy()
    if "_eval_group_name" in compat_df.columns and "_resolved_group_name" not in compat_df.columns:
        compat_df["_resolved_group_name"] = compat_df["_eval_group_name"]
    if "min_group_size" in inspect.signature(compute_fn).parameters:
        return compute_fn(compat_df, min_group_size=group_metric_min_size)
    return compute_fn(compat_df)


def build_variant_lookup(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    repo_root = os.path.abspath(args.external_repo_root)
    variants: Dict[str, Dict[str, Any]] = {}
    for spec in VARIANT_SPECS:
        key = spec["backend"].replace("-", "_")
        repo_arg = getattr(args, f"{key}_repo_dir", "") or os.path.join(repo_root, spec["default_repo_name"])
        env_arg = getattr(args, f"{key}_env_name", "") or spec["default_env_name"]
        variants[spec["name"]] = {
            **spec,
            "repo_dir": os.path.abspath(repo_arg),
            "conda_env_name": env_arg,
        }
        if spec["backend"] == "ma_agiqa":
            variants[spec["name"]]["tensor_root"] = os.path.abspath(args.ma_agiqa_tensor_root) if args.ma_agiqa_tensor_root else ""
        if spec["backend"] == "m3_agiqa":
            base_config = args.m3_agiqa_base_config or os.path.join(repo_arg, "cfg", "minicpm-xlstm-agiqa-3k.yaml")
            variants[spec["name"]].update(
                {
                    "base_config_path": os.path.abspath(base_config),
                    "train_pool_json": os.path.abspath(args.m3_agiqa_train_pool_json) if args.m3_agiqa_train_pool_json else "",
                    "eval_pool_json": os.path.abspath(args.m3_agiqa_eval_pool_json) if args.m3_agiqa_eval_pool_json else "",
                    "test_pool_json": os.path.abspath(args.m3_agiqa_test_pool_json) if args.m3_agiqa_test_pool_json else "",
                    "dataset_name": args.m3_agiqa_dataset_name,
                }
            )
    return variants


def validate_variant_runtime_inputs(selected_variants: Sequence[str], variant_lookup: Dict[str, Dict[str, Any]]) -> None:
    for variant_name in selected_variants:
        spec = variant_lookup[variant_name]
        if spec["backend"] == "ma_agiqa" and not spec.get("tensor_root"):
            raise ValueError("MA-AGIQA 需要传入 --ma_agiqa_tensor_root 指向预提取的 tensor cache。")
        if spec["backend"] == "m3_agiqa":
            missing = [
                key for key in ["train_pool_json", "eval_pool_json", "test_pool_json"]
                if not spec.get(key)
            ]
            if missing:
                raise ValueError(
                    "M3-AGIQA 需要传入全量 processed json 池路径："
                    "--m3_agiqa_train_pool_json、--m3_agiqa_eval_pool_json、--m3_agiqa_test_pool_json。"
                )


def default_variants_for_task(task: str, variant_lookup: Dict[str, Dict[str, Any]]) -> List[str]:
    return [
        name
        for name, spec in variant_lookup.items()
        if task in spec["supported_tasks"]
    ]


def build_run_artifact_paths(output_root: str, split_seed: int, variant_name: str, seed: int) -> Dict[str, str]:
    run_dir = os.path.join(os.path.abspath(output_root), f"split_{split_seed}", variant_name, f"seed_{seed}")
    return {
        "run_dir": run_dir,
        "preds_path": os.path.join(run_dir, "preds.csv"),
        "metrics_path": os.path.join(run_dir, "metrics.json"),
        "train_log_path": os.path.join(run_dir, "train_log.json"),
        "config_path": os.path.join(run_dir, "config.json"),
        "checkpoint_path": os.path.join(run_dir, "checkpoint.pt"),
        "adapter_metadata_path": os.path.join(run_dir, "adapter_metadata.json"),
        "val_preds_raw_path": os.path.join(run_dir, "val_preds_raw.csv"),
        "test_preds_raw_path": os.path.join(run_dir, "test_preds_raw.csv"),
        "train_csv_path": os.path.join(run_dir, "train.csv"),
        "val_csv_path": os.path.join(run_dir, "val.csv"),
        "test_csv_path": os.path.join(run_dir, "test.csv"),
    }


def is_completed_run(run_paths: Dict[str, str]) -> bool:
    required_keys = [
        "preds_path",
        "metrics_path",
        "train_log_path",
        "config_path",
        "checkpoint_path",
        "adapter_metadata_path",
    ]
    for key in required_keys:
        path = run_paths[key]
        if not os.path.exists(path) or os.path.getsize(path) <= 0:
            return False
    return True


def saved_run_matches_config(run_paths: Dict[str, str], config_payload: Dict[str, Any]) -> bool:
    if not os.path.exists(run_paths["config_path"]):
        return False
    with open(run_paths["config_path"], "r", encoding="utf-8") as f:
        saved_cfg = json.load(f)
    for key in RUN_CONFIG_MATCH_KEYS:
        if saved_cfg.get(key) != config_payload.get(key):
            return False
    return True


def load_completed_run_result(run_paths: Dict[str, str]) -> Dict[str, Any]:
    with open(run_paths["metrics_path"], "r", encoding="utf-8") as f:
        metrics = json.load(f)
    split_payload = None
    if os.path.exists(run_paths["train_log_path"]):
        with open(run_paths["train_log_path"], "r", encoding="utf-8") as f:
            split_payload = json.load(f).get("split")
    return {
        "run_dir": run_paths["run_dir"],
        "checkpoint_path": run_paths["checkpoint_path"],
        "preds_path": run_paths["preds_path"],
        "metrics_path": run_paths["metrics_path"],
        "train_log_path": run_paths["train_log_path"],
        "config_path": run_paths["config_path"],
        "adapter_metadata_path": run_paths["adapter_metadata_path"],
        "metrics": metrics,
        "split": split_payload,
    }


def build_prediction_frame(df: pd.DataFrame, pred_raw: Sequence[float], task: str) -> pd.DataFrame:
    label_column = label_column_for_task(task)
    pred_df = pd.DataFrame(
        {
            "name": df["name"].astype(str).tolist(),
            "prompt": df["prompt"].astype(str).tolist() if "prompt" in df.columns else [""] * len(df),
            "target": pd.to_numeric(df[label_column], errors="coerce").to_numpy(dtype="float64") / 5.0,
            "pred": pd.Series(pred_raw, dtype="float64").to_numpy(dtype="float64") / 5.0,
            "task": [task] * len(df),
        }
    )
    pred_df["pred"] = pred_df["pred"].clip(0.0, 1.0)
    meta_cols = [
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
    available_meta_cols = [col for col in meta_cols if col in df.columns]
    if available_meta_cols:
        pred_df = pd.concat([pred_df, df[available_meta_cols].reset_index(drop=True)], axis=1)
    return pred_df


def compute_task_metric_payload(
    pred_df: pd.DataFrame,
    bootstrap_iters: int,
    bootstrap_seed: int,
    group_metric_min_size: int,
) -> Dict[str, Any]:
    task_metrics = model_mod._metric_block(
        pred_df["target"].to_numpy(dtype="float64"),
        pred_df["pred"].to_numpy(dtype="float64"),
        bootstrap_iters,
        bootstrap_seed,
    )
    group_input = pred_df.rename(columns={"target": "target_c", "pred": "pred_c"})
    group_metrics, group_summary = compute_group_metrics_compat(group_input, group_metric_min_size)
    return {
        "task_metrics": task_metrics,
        "group_metrics": group_metrics,
        "group_summary": group_summary,
        "bootstrap": {"iters": bootstrap_iters, "seed": bootstrap_seed},
    }


def summarize_run(variant_name: str, task: str, split_seed: int, seed: int, result: Dict[str, Any]) -> Dict[str, Any]:
    metrics = result["metrics"]
    return {
        "variant": variant_name,
        "task": task,
        "split_seed": split_seed,
        "seed": seed,
        "run_dir": result["run_dir"],
        "preds_path": result["preds_path"],
        "metrics_path": result["metrics_path"],
        "selection_score": metrics["selection_score"],
        "task_srocc": metrics["task_metrics"]["srocc"]["value"],
        "task_plcc": metrics["task_metrics"]["plcc"]["value"],
        "task_rmse": metrics["task_metrics"]["rmse"]["value"],
        "task_mae": metrics["task_metrics"]["mae"]["value"],
        "worst_group_srocc": metrics["group_summary"]["worst_group_c_srocc"],
        "worst_group_rmse": metrics["group_summary"]["worst_group_c_rmse"],
        "mean_group_srocc": metrics["group_summary"]["mean_group_c_srocc"],
        "trainable_params": metrics.get("trainable_params", 0),
        "total_params": metrics.get("total_params", 0),
        "runtime_sec": metrics.get("runtime_sec", 0.0),
        "best_epoch": metrics.get("best_epoch", -1),
    }


def aggregate_metrics_rows(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    numeric_cols = [
        "selection_score",
        "task_srocc",
        "task_plcc",
        "task_rmse",
        "task_mae",
        "worst_group_srocc",
        "worst_group_rmse",
        "mean_group_srocc",
        "trainable_params",
        "total_params",
        "runtime_sec",
    ]
    agg = df.groupby(["task", "variant"], sort=False)[numeric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["task", "variant"] + [f"{name}_{stat}" for name, stat in agg.columns.tolist()[2:]]
    variant_order = {spec["name"]: idx for idx, spec in enumerate(VARIANT_SPECS)}
    agg["_variant_order"] = agg["variant"].map(lambda name: variant_order.get(name, len(variant_order)))
    return agg.sort_values(["task", "_variant_order", "variant"]).drop(columns=["_variant_order"]).reset_index(drop=True)


def flatten_group_metrics(run_rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for row in run_rows:
        with open(row["metrics_path"], "r", encoding="utf-8") as f:
            metrics = json.load(f)
        for group_metric in metrics.get("group_metrics", []):
            rows.append(
                {
                    "variant": row["variant"],
                    "task": row["task"],
                    "split_seed": row["split_seed"],
                    "seed": row["seed"],
                    **group_metric,
                }
            )
    return pd.DataFrame(rows)


def _fmt_mean_std(mean_value: float, std_value: float) -> str:
    std = 0.0 if pd.isna(std_value) else float(std_value)
    return f"{float(mean_value):.4f} +/- {std:.4f}"


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data_"
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(df.iloc[i][col]) for col in columns) + " |" for i in range(len(df))]
    return "\n".join([header, separator] + rows)


def write_report(
    output_root: str,
    task: str,
    train_seeds: Sequence[int],
    split_seeds: Sequence[int],
    aggregate_df: pd.DataFrame,
    split_description: str,
) -> None:
    source = aggregate_df.set_index("variant") if not aggregate_df.empty else pd.DataFrame()
    rows: List[Dict[str, str]] = []
    for spec in VARIANT_SPECS:
        variant = spec["name"]
        if aggregate_df.empty or variant not in source.index:
            continue
        row = source.loc[variant]
        rows.append(
            {
                "Variant": DISPLAY_NAME_BY_VARIANT.get(variant, variant),
                "SROCC": _fmt_mean_std(row["task_srocc_mean"], row["task_srocc_std"]),
                "PLCC": _fmt_mean_std(row["task_plcc_mean"], row["task_plcc_std"]),
                "RMSE": _fmt_mean_std(row["task_rmse_mean"], row["task_rmse_std"]),
                "MAE": _fmt_mean_std(row["task_mae_mean"], row["task_mae_std"]),
                "Worst-Group SROCC": _fmt_mean_std(row["worst_group_srocc_mean"], row["worst_group_srocc_std"]),
                "Runtime (s)": _fmt_mean_std(row["runtime_sec_mean"], row["runtime_sec_std"]),
            }
        )
    table_df = pd.DataFrame(rows)
    report_path = os.path.join(output_root, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# External Fair Rerun Report ({task})\n\n")
        f.write(f"- Split seeds: `{','.join(str(x) for x in split_seeds)}`\n")
        f.write(f"- Training seeds: `{','.join(str(x) for x in train_seeds)}`\n")
        f.write(f"- Task: `{task}`\n")
        f.write(f"- Split protocol: {split_description}\n")
        f.write("- Checkpoint selection: validation SROCC; if validation is disabled, use the no-validation stop rule configured for the adapter\n\n")
        f.write(_markdown_table(table_df))
        f.write("\n")


def _candidate_conda_roots() -> List[str]:
    roots: List[str] = []
    conda_exe = os.environ.get("CONDA_EXE", "").strip()
    if conda_exe:
        roots.append(os.path.dirname(os.path.dirname(os.path.abspath(conda_exe))))
    exec_path = os.path.abspath(sys.executable)
    marker = f"{os.sep}envs{os.sep}"
    if marker in exec_path:
        roots.append(exec_path.split(marker, 1)[0])
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if conda_prefix:
        prefix_abs = os.path.abspath(conda_prefix)
        if os.path.basename(os.path.dirname(prefix_abs)) == "envs":
            roots.append(os.path.dirname(os.path.dirname(prefix_abs)))
    unique_roots: List[str] = []
    seen = set()
    for root in roots:
        if root and root not in seen:
            seen.add(root)
            unique_roots.append(root)
    return unique_roots


def resolve_env_python(env_name: str) -> str:
    env_name = str(env_name).strip()
    if not env_name:
        return ""
    if os.path.isabs(env_name):
        candidate_prefixes = [env_name]
    else:
        candidate_prefixes = []
        for root in _candidate_conda_roots():
            if env_name == "base":
                candidate_prefixes.append(root)
            candidate_prefixes.append(os.path.join(root, "envs", env_name))

    python_names = ["python.exe", os.path.join("bin", "python")]
    for prefix in candidate_prefixes:
        for python_name in python_names:
            candidate = os.path.join(prefix, python_name)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
    return ""


def dispatch_external_adapter(command: List[str], workdir: Optional[str] = None) -> None:
    subprocess.run(command, check=True, cwd=workdir or None)


def log_run_status(
    index: int,
    total: int,
    status: str,
    variant_name: str,
    split_seed: int,
    seed: int,
    extra: str = "",
) -> None:
    message = f"[{index}/{total}] {status} {variant_name} (split={split_seed}, seed={seed})"
    if extra:
        message += f" {extra}"
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser("Run external fair-rerun baselines")
    parser.add_argument("--data_csv_path", required=True)
    parser.add_argument("--image_base_dir", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--task", default="quality", choices=["quality", "alignment"])
    parser.add_argument("--variants", default="")
    parser.add_argument("--device", default="cuda" if model_mod.torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bootstrap_iters", type=int, default=5000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument("--split_seeds", default="42,52,62")
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--train_seeds", default="11,22,33")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size_within_train", type=float, default=0.1)
    parser.add_argument("--use_existing_split_column", action="store_true")
    parser.add_argument("--split_column", default="split")
    parser.add_argument("--split_train_labels", default="train")
    parser.add_argument("--split_val_labels", default="val,valid,validation,dev")
    parser.add_argument("--split_test_labels", default="test")
    parser.add_argument("--no_validation_split", action="store_true")
    parser.add_argument("--no_refit_trainval", action="store_true")
    parser.add_argument("--train_loss_stop_threshold", type=float, default=None)
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--group_column", default="auto")
    parser.add_argument("--group_metric_min_size", type=int, default=3)
    parser.add_argument("--external_repo_root", default="_repo_inspect")
    parser.add_argument("--ipce_repo_dir", default="")
    parser.add_argument("--clip_agiqa_repo_dir", default="")
    parser.add_argument("--ma_agiqa_repo_dir", default="")
    parser.add_argument("--m3_agiqa_repo_dir", default="")
    parser.add_argument("--ipce_env_name", default="IPCE")
    parser.add_argument("--clip_agiqa_env_name", default="CLIP-AGIQA")
    parser.add_argument("--ma_agiqa_env_name", default="MA-AGIQA")
    parser.add_argument("--m3_agiqa_env_name", default="M3-AGIQA")
    parser.add_argument("--ma_agiqa_tensor_root", default="")
    parser.add_argument("--m3_agiqa_base_config", default="")
    parser.add_argument("--m3_agiqa_train_pool_json", default="")
    parser.add_argument("--m3_agiqa_eval_pool_json", default="")
    parser.add_argument("--m3_agiqa_test_pool_json", default="")
    parser.add_argument("--m3_agiqa_dataset_name", default="agiqa-3k")
    args = parser.parse_args()

    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)
    task = args.task
    label_column = label_column_for_task(task)

    variant_lookup = build_variant_lookup(args)
    if args.variants.strip():
        selected_variants = parse_variant_list(args.variants)
    else:
        selected_variants = default_variants_for_task(task, variant_lookup)
    if not selected_variants:
        raise ValueError(f"No external variants are available for task='{task}'.")
    unknown = [name for name in selected_variants if name not in variant_lookup]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    unsupported = [name for name in selected_variants if task not in variant_lookup[name]["supported_tasks"]]
    if unsupported:
        raise ValueError(f"Variants {unsupported} do not support task='{task}'.")
    validate_variant_runtime_inputs(selected_variants, variant_lookup)

    df = pd.read_csv(args.data_csv_path)
    df = model_mod.ensure_split_row_ids(df)
    df, split_metadata = annotate_split_context_compat(df, group_column=args.group_column)

    requested_split_seeds = [args.split_seed] if args.split_seed is not None else parse_seed_list(args.split_seeds)
    train_seeds = parse_seed_list(args.train_seeds)
    split_files: Dict[str, str] = {}
    if args.use_existing_split_column:
        split_seeds, split_files, split_description = ablation_mod.materialize_existing_split_files(
            output_root=output_root,
            data_csv_path=args.data_csv_path,
            split_column=args.split_column,
            group_column=args.group_column,
            val_size_within_train=args.val_size_within_train,
            requested_split_seeds=requested_split_seeds,
            disable_validation_split=args.no_validation_split,
            train_labels=[x.strip() for x in args.split_train_labels.split(",") if x.strip()],
            val_labels=[x.strip() for x in args.split_val_labels.split(",") if x.strip()],
            test_labels=[x.strip() for x in args.split_test_labels.split(",") if x.strip()],
        )
    else:
        split_seeds = requested_split_seeds
        split_description = (
            f"随机 outer 80/20 + train 内再切 validation={args.val_size_within_train:.2f}；"
            "同一 split_seed 复用到所有外部模型"
        )
        for split_seed in split_seeds:
            split_path = ablation_mod.build_split_file_path(output_root, split_seed)
            load_or_create_nested_split_compat(
                df=df,
                test_size=args.test_size,
                val_size_within_train=args.val_size_within_train,
                split_seed=split_seed,
                split_file=split_path,
                split_metadata=split_metadata,
                disable_validation_split=args.no_validation_split,
            )
            split_files[str(split_seed)] = split_path

    split_manifest = {
        "task": task,
        "label_column": label_column,
        "split_seeds": split_seeds,
        "train_seeds": train_seeds,
        "split_files": split_files,
    }
    with open(os.path.join(output_root, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, ensure_ascii=False, indent=2)

    adapter_script = os.path.join(os.path.abspath(os.path.dirname(__file__)), "external_baseline_adapter.py")
    run_rows: List[Dict[str, Any]] = []
    total_runs = len(split_seeds) * len(selected_variants) * len(train_seeds)
    current_run = 0

    for split_seed in split_seeds:
        split_path = split_files[str(split_seed)]
        train_ids, val_ids, test_ids, split_payload = load_or_create_nested_split_compat(
            df=df,
            test_size=args.test_size,
            val_size_within_train=args.val_size_within_train,
            split_seed=split_seed,
            split_file=split_path,
            split_metadata=split_metadata,
            disable_validation_split=args.no_validation_split,
        )
        train_df, val_df, test_df = model_mod.split_dataframe_from_three_way_ids(df, train_ids, val_ids, test_ids)
        for variant_name in selected_variants:
            variant_spec = variant_lookup[variant_name]
            for seed in train_seeds:
                current_run += 1
                run_paths = build_run_artifact_paths(output_root, split_seed, variant_name, seed)
                config_payload = {
                    "variant": variant_name,
                    "backend": variant_spec["backend"],
                    "task": task,
                    "label_column": label_column,
                    "repo_dir": variant_spec["repo_dir"],
                    "conda_env_name": variant_spec["conda_env_name"],
                    "data_csv_path": os.path.abspath(args.data_csv_path),
                    "image_base_dir": os.path.abspath(args.image_base_dir),
                    "device": args.device,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "num_workers": args.num_workers,
                    "test_size": args.test_size,
                    "val_size_within_train": args.val_size_within_train,
                    "no_validation_split": bool(args.no_validation_split),
                    "train_loss_stop_threshold": args.train_loss_stop_threshold,
                    "split_seed": split_seed,
                    "seed": seed,
                    "split_file": split_path,
                    "refit_on_trainval": not args.no_refit_trainval,
                    "tensor_root": variant_spec.get("tensor_root", ""),
                    "base_config_path": variant_spec.get("base_config_path", ""),
                    "train_pool_json": variant_spec.get("train_pool_json", ""),
                    "eval_pool_json": variant_spec.get("eval_pool_json", ""),
                    "test_pool_json": variant_spec.get("test_pool_json", ""),
                    "dataset_name": variant_spec.get("dataset_name", ""),
                }
                if not args.force_rerun and is_completed_run(run_paths) and saved_run_matches_config(run_paths, config_payload):
                    log_run_status(current_run, total_runs, "Skip", variant_name, split_seed, seed)
                    run_rows.append(summarize_run(variant_name, task, split_seed, seed, load_completed_run_result(run_paths)))
                    continue

                log_run_status(current_run, total_runs, "Start", variant_name, split_seed, seed)
                os.makedirs(run_paths["run_dir"], exist_ok=True)
                train_df.to_csv(run_paths["train_csv_path"], index=False, encoding="utf-8")
                val_df.to_csv(run_paths["val_csv_path"], index=False, encoding="utf-8")
                test_df.to_csv(run_paths["test_csv_path"], index=False, encoding="utf-8")

                adapter_python = resolve_env_python(variant_spec["conda_env_name"])
                if adapter_python:
                    cmd = [adapter_python, adapter_script]
                else:
                    cmd = [
                        "conda", "run", "--no-capture-output", "-n", variant_spec["conda_env_name"],
                        "python", adapter_script,
                    ]
                cmd += [
                    "--backend", variant_spec["backend"],
                    "--repo_dir", variant_spec["repo_dir"],
                    "--train_csv", run_paths["train_csv_path"],
                    "--val_csv", run_paths["val_csv_path"],
                    "--test_csv", run_paths["test_csv_path"],
                    "--image_base_dir", os.path.abspath(args.image_base_dir),
                    "--output_dir", run_paths["run_dir"],
                    "--task", task,
                    "--label_column", label_column,
                    "--device", args.device,
                    "--seed", str(seed),
                    "--num_workers", str(args.num_workers),
                ]
                if args.epochs is not None:
                    cmd += ["--epochs", str(args.epochs)]
                if args.batch_size is not None:
                    cmd += ["--batch_size", str(args.batch_size)]
                if args.lr is not None:
                    cmd += ["--lr", str(args.lr)]
                if args.train_loss_stop_threshold is not None:
                    cmd += ["--train_loss_stop_threshold", str(args.train_loss_stop_threshold)]
                if not args.no_refit_trainval:
                    cmd.append("--refit_on_trainval")
                if variant_spec["backend"] == "ma_agiqa":
                    cmd += ["--tensor_root", variant_spec["tensor_root"]]
                if variant_spec["backend"] == "m3_agiqa":
                    cmd += [
                        "--base_config", variant_spec["base_config_path"],
                        "--train_pool_json", variant_spec["train_pool_json"],
                        "--eval_pool_json", variant_spec["eval_pool_json"],
                        "--test_pool_json", variant_spec["test_pool_json"],
                        "--dataset_name", variant_spec["dataset_name"],
                    ]
                try:
                    dispatch_external_adapter(cmd, workdir=variant_spec["repo_dir"])
                except Exception as exc:
                    log_run_status(current_run, total_runs, "Failed", variant_name, split_seed, seed, extra=f"error={exc}")
                    raise

                val_raw_df = pd.read_csv(run_paths["val_preds_raw_path"])
                test_raw_df = pd.read_csv(run_paths["test_preds_raw_path"])
                if len(val_raw_df) != len(val_df) or len(test_raw_df) != len(test_df):
                    raise ValueError(f"Adapter output row count mismatch for {variant_name} split={split_seed} seed={seed}.")

                val_pred_df = build_prediction_frame(val_df, val_raw_df["pred_raw"].tolist(), task)
                test_pred_df = build_prediction_frame(test_df, test_raw_df["pred_raw"].tolist(), task)
                val_payload = compute_task_metric_payload(
                    val_pred_df,
                    bootstrap_iters=args.bootstrap_iters,
                    bootstrap_seed=args.bootstrap_seed,
                    group_metric_min_size=args.group_metric_min_size,
                )
                final_payload = compute_task_metric_payload(
                    test_pred_df,
                    bootstrap_iters=args.bootstrap_iters,
                    bootstrap_seed=args.bootstrap_seed,
                    group_metric_min_size=args.group_metric_min_size,
                )
                with open(run_paths["adapter_metadata_path"], "r", encoding="utf-8") as f:
                    adapter_meta = json.load(f)
                selection_score = adapter_meta.get("selection_score", val_payload["task_metrics"]["srocc"]["value"])
                selection_based_on = adapter_meta.get("selection_based_on", "validation")
                selection_metric = adapter_meta.get("selection_metric", "srocc")
                final_payload.update(
                    {
                        "task": task,
                        "selection_score": float(selection_score),
                        "validation_metrics": val_payload,
                        "best_epoch": int(adapter_meta.get("best_epoch", 1)),
                        "trainable_params": int(adapter_meta.get("trainable_params", 0)),
                        "total_params": int(adapter_meta.get("total_params", 0)),
                        "runtime_sec": float(adapter_meta.get("runtime_sec", 0.0)),
                        "split_seed": int(split_seed),
                        "seed": int(seed),
                        "split_file": split_path,
                        "evaluation_split": "test",
                        "selection": {
                            "metric": str(selection_metric),
                            "based_on": str(selection_based_on),
                            "best_epoch": int(adapter_meta.get("best_epoch", 1)),
                            "refit_trainval": bool(adapter_meta.get("refit_on_trainval", False)),
                            "train_loss_stop_threshold": adapter_meta.get("train_loss_stop_threshold"),
                            "stopped_on_train_loss_threshold": bool(
                                adapter_meta.get("stopped_on_train_loss_threshold", False)
                            ),
                        },
                    }
                )
                with open(run_paths["metrics_path"], "w", encoding="utf-8") as f:
                    json.dump(final_payload, f, ensure_ascii=False, indent=2)
                with open(run_paths["train_log_path"], "w", encoding="utf-8") as f:
                    json.dump({"split": split_payload, "adapter": adapter_meta}, f, ensure_ascii=False, indent=2)
                with open(run_paths["config_path"], "w", encoding="utf-8") as f:
                    json.dump(config_payload, f, ensure_ascii=False, indent=2)
                test_pred_df.to_csv(run_paths["preds_path"], index=False, encoding="utf-8")
                log_run_status(
                    current_run,
                    total_runs,
                    "Done",
                    variant_name,
                    split_seed,
                    seed,
                    extra=f"run_dir={run_paths['run_dir']}",
                )
                run_rows.append(summarize_run(variant_name, task, split_seed, seed, load_completed_run_result(run_paths)))

    metrics_df = pd.DataFrame(run_rows)
    aggregate_df = aggregate_metrics_rows(run_rows)
    group_df = flatten_group_metrics(run_rows)
    metrics_df.to_csv(os.path.join(output_root, "metrics_by_seed.csv"), index=False, encoding="utf-8")
    aggregate_df.to_csv(os.path.join(output_root, "aggregate.csv"), index=False, encoding="utf-8")
    group_df.to_csv(os.path.join(output_root, "group_metrics.csv"), index=False, encoding="utf-8")
    write_report(output_root, task, train_seeds, split_seeds, aggregate_df, split_description)


if __name__ == "__main__":
    main()
