#!/usr/bin/env python3
"""Run a local VLM as a direct judge for perceptual quality and prompt alignment."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

try:
    from openai import OpenAI

    _OPENAI_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - import error depends on local env
    OpenAI = None  # type: ignore[assignment]
    _OPENAI_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


SYSTEM_PROMPT = """
You are an expert judge for text-to-image evaluation.

You will receive:
1. one image
2. the original text prompt

You must score two dimensions independently:
- perceptual_quality_score: visual fidelity, naturalness, artifact severity, structural coherence, readability, and overall perceptual quality. Ignore whether the prompt was followed.
- alignment_score: how well the image matches the prompt semantically. Ignore pure aesthetics unless they directly obscure semantic matching.

Scoring rules:
- Use continuous scores in [0, 5].
- 5 means excellent / fully matched.
- 0 means failed / completely mismatched.
- Be strict and consistent.

Return JSON only. Do not use markdown.
Return exactly:
{
  "perceptual_quality_score": 0.0,
  "alignment_score": 0.0,
  "quality_confidence": 0.0,
  "alignment_confidence": 0.0,
  "quality_reason": "",
  "alignment_reason": ""
}
""".strip()


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def safe_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def guess_mime_type(path: str) -> str:
    ext = os.path.splitext(path.lower())[1]
    if ext == ".png":
        return "image/png"
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"


def image_to_data_url(image_path: str) -> str:
    mime = guess_mime_type(image_path)
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def extract_json_object(text: Any) -> Dict[str, Any]:
    content = safe_str(text)
    if not content:
        raise ValueError("Empty model response.")

    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    cleaned = re.sub(r"^```(?:json)?", "", content, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj

    raise ValueError(f"Failed to parse JSON from model output: {content[:500]}")


def infer_score_scale(values: Sequence[Any]) -> Tuple[float, str]:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=np.float64)
    if arr.size == 0:
        return 1.0, "empty_default"
    max_abs = float(np.max(np.abs(arr)))
    if max_abs <= 1.05:
        return 1.0, "unit_interval"
    if max_abs <= 5.05:
        return 5.0, "five_point"
    if max_abs <= 100.5:
        return 100.0, "percent"
    return max_abs, "max_abs"


def normalize_by_scale(values: Sequence[Any], scale: float) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=np.float64)
    if scale <= 0:
        return arr
    return np.clip(arr / float(scale), 0.0, 1.0)


def safe_srocc(target: Sequence[Any], pred: Sequence[Any]) -> float:
    y = pd.to_numeric(pd.Series(list(target)), errors="coerce").to_numpy(dtype=np.float64)
    yhat = pd.to_numeric(pd.Series(list(pred)), errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if int(mask.sum()) < 2:
        return 0.0
    value = spearmanr(y[mask], yhat[mask]).correlation
    return 0.0 if value is None or not math.isfinite(float(value)) else float(value)


def safe_plcc(target: Sequence[Any], pred: Sequence[Any]) -> float:
    y = pd.to_numeric(pd.Series(list(target)), errors="coerce").to_numpy(dtype=np.float64)
    yhat = pd.to_numeric(pd.Series(list(pred)), errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if int(mask.sum()) < 2:
        return 0.0
    value = pearsonr(y[mask], yhat[mask])[0]
    return 0.0 if value is None or not math.isfinite(float(value)) else float(value)


def build_metric_block(target_raw: Sequence[Any], pred_raw: Sequence[Any]) -> Dict[str, Any]:
    target_scale, target_norm_name = infer_score_scale(target_raw)
    pred_scale, pred_norm_name = infer_score_scale(pred_raw)
    target_norm = normalize_by_scale(target_raw, target_scale)
    pred_norm = normalize_by_scale(pred_raw, pred_scale)
    valid = np.isfinite(target_norm) & np.isfinite(pred_norm)
    return {
        "count": int(valid.sum()),
        "srocc": safe_srocc(target_norm, pred_norm),
        "plcc": safe_plcc(target_norm, pred_norm),
        "target_scale": float(target_scale),
        "target_scale_name": target_norm_name,
        "prediction_scale": float(pred_scale),
        "prediction_scale_name": pred_norm_name,
    }


def ensure_split_row_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    out["_split_row_id"] = np.arange(len(out), dtype=np.int64)
    return out


def filter_dataframe_by_split_file(
    df: pd.DataFrame,
    split_file: str,
    split_role: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    prepared = ensure_split_row_ids(df)
    if not split_file:
        return prepared, {"enabled": False, "split_role": "all", "num_rows": int(len(prepared))}

    with open(split_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if split_role == "all":
        return prepared, {
            "enabled": True,
            "split_role": "all",
            "num_rows": int(len(prepared)),
            "split_file": split_file,
        }

    if split_role not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split_role='{split_role}'.")

    if split_role == "test" and "test_ids" not in payload:
        ids = payload.get("val_ids", [])
        resolved_role = "legacy_test_from_val_ids"
    else:
        key = f"{split_role}_ids"
        ids = payload.get(key, [])
        resolved_role = split_role

    id_set = {int(x) for x in ids}
    filtered = prepared[prepared["_split_row_id"].isin(id_set)].sort_values("_split_row_id").reset_index(drop=True)
    return filtered, {
        "enabled": True,
        "split_role": resolved_role,
        "split_file": split_file,
        "num_rows": int(len(filtered)),
    }


def make_row_key(idx: int, row: pd.Series, image_column: str, prompt_column: str) -> str:
    return f"{idx}||{safe_str(row.get(image_column))}||{safe_str(row.get(prompt_column))}"


def find_image_path(image_root: str, image_ref: str) -> Optional[str]:
    if not image_ref:
        return None

    candidates: List[str] = []
    if os.path.isabs(image_ref):
        candidates.append(image_ref)
    else:
        candidates.append(os.path.join(image_root, image_ref) if image_root else image_ref)

    stem, ext = os.path.splitext(image_ref)
    if ext:
        stem_name = stem
    else:
        stem_name = image_ref
    for suffix in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"]:
        if os.path.isabs(stem_name):
            candidates.append(stem_name + suffix)
        else:
            candidates.append(os.path.join(image_root, stem_name + suffix) if image_root else stem_name + suffix)

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def build_user_text(prompt: str) -> str:
    payload = {
        "task": "Score the image on perceptual quality and text-image alignment.",
        "prompt": prompt,
        "rubric": {
            "perceptual_quality_score": "0-5 continuous; judge image quality only",
            "alignment_score": "0-5 continuous; judge prompt-image semantic match",
            "quality_confidence": "0-1 confidence",
            "alignment_confidence": "0-1 confidence",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _first_present(obj: Dict[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if name in obj:
            return obj.get(name)
    return default


def postprocess_score_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    quality = safe_float(
        _first_present(
            obj,
            [
                "perceptual_quality_score",
                "quality_score",
                "vlm_quality_score",
                "judge_quality_score",
                "quality",
            ],
        ),
        default=float("nan"),
    )
    alignment = safe_float(
        _first_present(
            obj,
            [
                "alignment_score",
                "align_score",
                "judge_align_score",
                "judge_alignment_score",
                "consistency_score",
                "alignment",
            ],
        ),
        default=float("nan"),
    )
    quality_conf = safe_float(
        _first_present(obj, ["quality_confidence", "perceptual_quality_confidence", "confidence"], 0.0),
        default=0.0,
    )
    alignment_conf = safe_float(
        _first_present(obj, ["alignment_confidence", "align_confidence", "confidence"], 0.0),
        default=0.0,
    )

    return {
        "judge_quality_score": clamp(quality, 0.0, 5.0) if math.isfinite(quality) else float("nan"),
        "judge_alignment_score": clamp(alignment, 0.0, 5.0) if math.isfinite(alignment) else float("nan"),
        "quality_confidence": clamp(quality_conf, 0.0, 1.0),
        "alignment_confidence": clamp(alignment_conf, 0.0, 1.0),
        "quality_reason": safe_str(_first_present(obj, ["quality_reason", "perceptual_quality_reason"])),
        "alignment_reason": safe_str(_first_present(obj, ["alignment_reason", "align_reason"])),
    }


class LocalVLMJudge:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "llama",
        temperature: float = 0.0,
        max_retries: int = 5,
        timeout: float = 600.0,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError(
                "openai package is required for local VLM judging. "
                f"Import error: {_OPENAI_IMPORT_ERROR or 'unknown error'}"
            )
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self._local = threading.local()

    def _get_client(self) -> OpenAI:
        client = getattr(self._local, "client", None)
        if client is None:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            self._local.client = client
        return client

    def judge_one(self, image_path: str, prompt: str) -> Dict[str, Any]:
        data_url = image_to_data_url(image_path)
        user_text = build_user_text(prompt)

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                client = self._get_client()
                try:
                    resp = client.chat.completions.create(
                        model=self.model,
                        temperature=self.temperature,
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": user_text},
                                    {"type": "image_url", "image_url": {"url": data_url}},
                                ],
                            },
                        ],
                    )
                except Exception as exc:
                    if "response_format" not in str(exc):
                        raise
                    resp = client.chat.completions.create(
                        model=self.model,
                        temperature=self.temperature,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": user_text},
                                    {"type": "image_url", "image_url": {"url": data_url}},
                                ],
                            },
                        ],
                    )

                if not resp.choices or not resp.choices[0].message.content:
                    raise ValueError("Empty response received from local VLM.")

                content = resp.choices[0].message.content
                if isinstance(content, list):
                    content = "\n".join(
                        safe_str(part.get("text")) if isinstance(part, dict) else safe_str(part)
                        for part in content
                    )
                obj = extract_json_object(content)
                return postprocess_score_output(obj)
            except Exception as exc:
                last_err = exc
                tqdm.write(
                    f"[Warning] Judge failed for {os.path.basename(image_path)} "
                    f"(attempt {attempt}/{self.max_retries}): {safe_str(exc)[:180]}"
                )
                if attempt < self.max_retries:
                    time.sleep(3 * attempt)
                else:
                    raise last_err

        raise RuntimeError("Unreachable state in judge_one.")


def is_connection_error_message(value: Any) -> bool:
    return "connection error" in safe_str(value).lower()


def record_requires_retry(record: Dict[str, Any]) -> bool:
    for field_name in (
        "_error",
        "error",
        "judge_quality_score",
        "judge_alignment_score",
        "quality_reason",
        "alignment_reason",
    ):
        if is_connection_error_message(record.get(field_name)):
            return True
    return False


def load_existing_jsonl_records(output_jsonl: str) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    if not output_jsonl or not os.path.exists(output_jsonl):
        return records

    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            row_key = safe_str(obj.get("_row_key"))
            if not row_key:
                continue
            records[row_key] = obj
    return records


def load_existing_csv_records(
    output_csv: str,
    source_df: pd.DataFrame,
    image_column: str,
    prompt_column: str,
) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    if not output_csv or not os.path.exists(output_csv):
        return records

    try:
        existing_df = pd.read_csv(output_csv)
    except Exception as exc:
        tqdm.write(f"[Warning] Failed to load existing predictions CSV: {safe_str(exc)[:180]}")
        return records

    for _, row in existing_df.iterrows():
        rec = row.to_dict()
        row_key = safe_str(rec.get("_row_key"))
        if not row_key:
            row_index = safe_float(rec.get("_row_index"), default=float("nan"))
            if math.isfinite(row_index):
                resolved_index = int(row_index)
                if 0 <= resolved_index < len(source_df):
                    row_key = make_row_key(
                        resolved_index,
                        source_df.iloc[resolved_index],
                        image_column,
                        prompt_column,
                    )
                    rec["_row_index"] = resolved_index
        if not row_key:
            continue
        rec["_row_key"] = row_key
        records[row_key] = rec

    return records


def load_existing_records(
    output_jsonl: str,
    output_csv: str,
    source_df: pd.DataFrame,
    image_column: str,
    prompt_column: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    jsonl_records = load_existing_jsonl_records(output_jsonl)
    csv_records = load_existing_csv_records(output_csv, source_df, image_column, prompt_column)

    merged_records = dict(jsonl_records)
    csv_only_count = 0
    for row_key, record in csv_records.items():
        current_record = merged_records.get(row_key)
        if current_record is None:
            merged_records[row_key] = record
            csv_only_count += 1
            continue
        if record_requires_retry(current_record) and not record_requires_retry(record):
            merged_records[row_key] = record

    return merged_records, {
        "jsonl_records": int(len(jsonl_records)),
        "csv_records": int(len(csv_records)),
        "csv_only_records": int(csv_only_count),
        "merged_records": int(len(merged_records)),
    }


def write_metrics_json(
    df: pd.DataFrame,
    metrics_path: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    metrics = {
        "quality": build_metric_block(df["target_quality_raw"].tolist(), df["judge_quality_score"].tolist()),
        "alignment": build_metric_block(df["target_alignment_raw"].tolist(), df["judge_alignment_score"].tolist()),
        "meta": metadata,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def process_all(args: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, "vlm_judge_raw.jsonl")
    output_csv = os.path.join(args.output_dir, "vlm_judge_predictions.csv")
    metrics_json = os.path.join(args.output_dir, "vlm_judge_metrics.json")

    df = pd.read_csv(args.data_csv_path)
    df, split_meta = filter_dataframe_by_split_file(df, args.split_file, args.split_role)
    if args.limit > 0:
        df = df.head(int(args.limit)).copy()

    judge = LocalVLMJudge(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_retries=args.max_retries,
        timeout=args.timeout,
    )
    existing_records: Dict[str, Dict[str, Any]]
    existing_stats: Dict[str, int]
    if args.overwrite:
        existing_records = {}
        existing_stats = {"jsonl_records": 0, "csv_records": 0, "csv_only_records": 0, "merged_records": 0}
    else:
        existing_records, existing_stats = load_existing_records(
            output_jsonl,
            output_csv,
            df,
            args.image_column,
            args.prompt_column,
        )
    file_lock = threading.Lock()
    records: List[Dict[str, Any]] = []
    tasks = []
    kept_existing_count = 0
    retry_connection_count = 0
    new_row_count = 0

    if args.overwrite and os.path.exists(output_jsonl):
        os.remove(output_jsonl)

    with open(output_jsonl, "a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=max(int(args.max_workers), 1)) as executor:
            for idx, row in df.iterrows():
                row_key = make_row_key(idx, row, args.image_column, args.prompt_column)
                existing_record = existing_records.get(row_key)
                if existing_record is not None and not record_requires_retry(existing_record):
                    kept_existing_count += 1
                    continue
                if existing_record is not None:
                    retry_connection_count += 1
                else:
                    new_row_count += 1

                image_ref = safe_str(row.get(args.image_column))
                prompt = safe_str(row.get(args.prompt_column))
                image_path = find_image_path(args.image_base_dir, image_ref)
                if image_path is None:
                    rec = {
                        "_row_index": int(idx),
                        "_row_key": row_key,
                        "name": image_ref,
                        "prompt": prompt,
                        "image_path": "",
                        "judge_quality_score": float("nan"),
                        "judge_alignment_score": float("nan"),
                        "quality_confidence": 0.0,
                        "alignment_confidence": 0.0,
                        "quality_reason": "",
                        "alignment_reason": "",
                        "_error": "image_not_found",
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    records.append(rec)
                    continue

                future = executor.submit(judge.judge_one, image_path, prompt)
                tasks.append((idx, row, row_key, image_path, future))

            print(
                "[Resume] "
                f"existing={existing_stats['merged_records']} "
                f"(jsonl={existing_stats['jsonl_records']}, csv={existing_stats['csv_records']}, csv_only={existing_stats['csv_only_records']}), "
                f"kept={kept_existing_count}, retry_connection_error={retry_connection_count}, new={new_row_count}"
            )

            future_to_payload = {future: (idx, row, row_key, image_path) for idx, row, row_key, image_path, future in tasks}
            for future in tqdm(as_completed(future_to_payload), total=len(future_to_payload), desc="Judging with local VLM"):
                idx, row, row_key, image_path = future_to_payload[future]
                try:
                    obj = future.result()
                    rec = {
                        "_row_index": int(idx),
                        "_row_key": row_key,
                        "name": safe_str(row.get(args.image_column)),
                        "prompt": safe_str(row.get(args.prompt_column)),
                        "image_path": image_path,
                        **obj,
                    }
                except Exception as exc:
                    rec = {
                        "_row_index": int(idx),
                        "_row_key": row_key,
                        "name": safe_str(row.get(args.image_column)),
                        "prompt": safe_str(row.get(args.prompt_column)),
                        "image_path": image_path,
                        "judge_quality_score": float("nan"),
                        "judge_alignment_score": float("nan"),
                        "quality_confidence": 0.0,
                        "alignment_confidence": 0.0,
                        "quality_reason": "",
                        "alignment_reason": "",
                        "_error": safe_str(exc),
                    }

                with file_lock:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                records.append(rec)

    merged: Dict[str, Dict[str, Any]] = {}
    for row_key, obj in existing_records.items():
        merged[row_key] = obj
    for rec in records:
        merged[safe_str(rec.get("_row_key"))] = rec

    merged_rows = [merged[make_row_key(idx, row, args.image_column, args.prompt_column)] for idx, row in df.iterrows()]
    pred_df = pd.DataFrame(merged_rows)
    target_df = df.reset_index(drop=True).copy()
    pred_df = pred_df.sort_values("_row_index").reset_index(drop=True)
    target_df = target_df.sort_values("_split_row_id").reset_index(drop=True) if "_split_row_id" in target_df.columns else target_df

    pred_df["target_quality_raw"] = pd.to_numeric(target_df[args.quality_column], errors="coerce")
    pred_df["target_alignment_raw"] = pd.to_numeric(target_df[args.alignment_column], errors="coerce")

    quality_target_scale, quality_target_scale_name = infer_score_scale(pred_df["target_quality_raw"].tolist())
    alignment_target_scale, alignment_target_scale_name = infer_score_scale(pred_df["target_alignment_raw"].tolist())
    quality_pred_scale, quality_pred_scale_name = infer_score_scale(pred_df["judge_quality_score"].tolist())
    alignment_pred_scale, alignment_pred_scale_name = infer_score_scale(pred_df["judge_alignment_score"].tolist())

    pred_df["target_quality_norm"] = normalize_by_scale(pred_df["target_quality_raw"].tolist(), quality_target_scale)
    pred_df["target_alignment_norm"] = normalize_by_scale(pred_df["target_alignment_raw"].tolist(), alignment_target_scale)
    pred_df["judge_quality_norm"] = normalize_by_scale(pred_df["judge_quality_score"].tolist(), quality_pred_scale)
    pred_df["judge_alignment_norm"] = normalize_by_scale(pred_df["judge_alignment_score"].tolist(), alignment_pred_scale)

    pred_df.to_csv(output_csv, index=False, encoding="utf-8")

    metrics = write_metrics_json(
        pred_df,
        metrics_json,
        metadata={
            "data_csv_path": args.data_csv_path,
            "image_base_dir": args.image_base_dir,
            "model": args.model,
            "base_url": args.base_url,
            "evaluated_rows": int(len(pred_df)),
            "quality_column": args.quality_column,
            "alignment_column": args.alignment_column,
            "quality_target_scale_name": quality_target_scale_name,
            "alignment_target_scale_name": alignment_target_scale_name,
            "quality_prediction_scale_name": quality_pred_scale_name,
            "alignment_prediction_scale_name": alignment_pred_scale_name,
            "split": split_meta,
        },
    )

    return {
        "jsonl_path": output_jsonl,
        "predictions_path": output_csv,
        "metrics_path": metrics_json,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser("Run a local VLM judge for quality/alignment scoring")
    parser.add_argument("--data_csv_path", required=True)
    parser.add_argument("--image_base_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--api_key", default="llama")
    parser.add_argument("--prompt_column", default="prompt")
    parser.add_argument("--image_column", default="name")
    parser.add_argument("--quality_column", default="mos_quality")
    parser.add_argument("--alignment_column", default="mos_align")
    parser.add_argument("--split_file", default="")
    parser.add_argument("--split_role", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = process_all(args)
    print(f"[OK] raw records: {result['jsonl_path']}")
    print(f"[OK] predictions: {result['predictions_path']}")
    print(f"[OK] metrics: {result['metrics_path']}")
    print(
        f"[Metrics] Q-SROCC={result['metrics']['quality']['srocc']:.4f}, "
        f"Q-PLCC={result['metrics']['quality']['plcc']:.4f}, "
        f"C-SROCC={result['metrics']['alignment']['srocc']:.4f}, "
        f"C-PLCC={result['metrics']['alignment']['plcc']:.4f}"
    )


if __name__ == "__main__":
    main()
