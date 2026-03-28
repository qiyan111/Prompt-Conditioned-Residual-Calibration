#!/usr/bin/env python3
"""Utilities for the Three-Stage Funnel Evaluator."""

from __future__ import annotations

import ast
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter


FUNNEL_CACHE_FIELDS = [
    "main_visual_subject",
    "focus_image_path",
    "focus_mask_path",
    "focus_valid",
    "focus_bbox_xyxy",
    "focus_area_ratio",
    "focus_center_x",
    "focus_center_y",
    "subject_match",
    "present_ratio",
    "missing_ratio",
    "uncertainty_ratio",
    "off_topic_ratio",
    "attribute_match_rate",
    "scene_match_rate",
    "style_match_rate",
    "relation_match_rate",
    "count_match_rate",
    "contradiction_flag",
    "confidence",
]

FUNNEL_LOGIC_FEATURE_NAMES = [
    "subject_match",
    "present_ratio",
    "missing_ratio",
    "off_topic_ratio",
    "attribute_match_rate",
    "scene_match_rate",
    "style_match_rate",
    "relation_match_rate",
    "count_match_rate",
    "confidence",
    "uncertainty_ratio",
    "contradiction_flag",
]

FUNNEL_LOGIC_ALIASES = {
    "uncertainty_ratio": ["uncertainty_ratio", "uncertain_ratio"],
    "attribute_match_rate": ["attribute_match_rate", "attr_true_ratio"],
    "scene_match_rate": ["scene_match_rate", "scene_true_ratio"],
    "style_match_rate": ["style_match_rate", "style_true_ratio"],
    "relation_match_rate": ["relation_match_rate", "relation_true_ratio"],
    "count_match_rate": ["count_match_rate", "count_true_ratio"],
    "contradiction_flag": ["contradiction_flag", "subject_hard_fail"],
}

FUNNEL_LOGIC_INDEX = {name: i for i, name in enumerate(FUNNEL_LOGIC_FEATURE_NAMES)}


def safe_text(value: Any) -> str:
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


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def has_value(record: Dict[str, Any], key: str) -> bool:
    if key not in record:
        return False
    value = record.get(key)
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    return True


def resolve_first_present(record: Dict[str, Any], names: Sequence[str], default: float = 0.0) -> float:
    for name in names:
        if has_value(record, name):
            return clamp01(safe_float(record.get(name), default))
    return clamp01(default)


def parse_json_like(value: Any, default: Any):
    if value is None:
        return default
    if isinstance(value, (dict, list, tuple)):
        return value
    text = safe_text(value)
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return default


def parse_listish(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    elif isinstance(value, tuple):
        items = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text or text in {"[]", "{}"}:
            return []
        parsed = parse_json_like(text, None)
        if isinstance(parsed, list):
            items = parsed
        else:
            items = text.split("|")
    else:
        items = [value]
    out: List[str] = []
    for item in items:
        txt = safe_text(item)
        if txt:
            out.append(txt)
    return out


def parse_bbox_xyxy(value: Any) -> Optional[Tuple[float, float, float, float]]:
    parsed = parse_json_like(value, None)
    if isinstance(parsed, dict):
        parsed = parsed.get("xyxy") or parsed.get("bbox") or parsed.get("box")
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in parsed]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def bbox_features_from_xyxy(
        bbox_xyxy: Optional[Tuple[float, float, float, float]],
        area_ratio: Optional[float] = None,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None) -> Dict[str, float]:
    if bbox_xyxy is None:
        return {
            "bbox_area": clamp01(safe_float(area_ratio, 0.0)),
            "bbox_center_x": clamp01(safe_float(center_x, 0.0)),
            "bbox_center_y": clamp01(safe_float(center_y, 0.0)),
            "bbox_width": 0.0,
            "bbox_height": 0.0,
        }

    x1, y1, x2, y2 = bbox_xyxy
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    if area_ratio is None:
        area_ratio = width * height
    if center_x is None:
        center_x = (x1 + x2) * 0.5
    if center_y is None:
        center_y = (y1 + y2) * 0.5
    return {
        "bbox_area": clamp01(safe_float(area_ratio, width * height)),
        "bbox_center_x": clamp01(safe_float(center_x, (x1 + x2) * 0.5)),
        "bbox_center_y": clamp01(safe_float(center_y, (y1 + y2) * 0.5)),
        "bbox_width": clamp01(width),
        "bbox_height": clamp01(height),
    }


def logic_dict_from_record(record: Dict[str, Any]) -> Dict[str, float]:
    subject_match_raw = record.get("subject_match", None)
    subject_match = clamp01(safe_float(subject_match_raw, 0.5))
    subject_valid = 1.0 if (
        subject_match_raw is not None and safe_text(subject_match_raw) != ""
    ) or safe_text(record.get("main_visual_subject")) else 0.0
    confidence = clamp01(safe_float(record.get("confidence"), 0.0))
    ratios = _compute_ratio_from_list_fields(record)
    present_ratio = resolve_first_present(record, ["present_ratio"], default=ratios["present_ratio"])
    missing_ratio = resolve_first_present(record, ["missing_ratio"], default=ratios["missing_ratio"])
    uncertainty_ratio = resolve_first_present(
        record,
        ["uncertainty_ratio", "uncertain_ratio"],
        default=ratios["uncertainty_ratio"],
    )
    off_topic_ratio = resolve_first_present(record, ["off_topic_ratio"], default=ratios["off_topic_ratio"])

    attr_true_ratio, attr_false_ratio = _mapping_true_false_ratio(
        record.get("attribute_requirements", record.get("attribute_match"))
    )
    scene_true_ratio, scene_false_ratio = _mapping_true_false_ratio(
        record.get("scene_requirements", record.get("scene_match"))
    )
    style_true_ratio, style_false_ratio = _mapping_true_false_ratio(
        record.get("style_requirements", record.get("style_match"))
    )
    relation_true_ratio, relation_false_ratio = _mapping_true_false_ratio(
        record.get("relation_requirements", record.get("relation_match"))
    )
    count_true_ratio, count_false_ratio = _mapping_true_false_ratio(
        record.get("count_requirements", record.get("count_match"))
    )

    attribute_match_rate = resolve_first_present(
        record,
        ["attribute_match_rate", "attr_true_ratio"],
        default=attr_true_ratio,
    )
    scene_match_rate = resolve_first_present(
        record,
        ["scene_match_rate", "scene_true_ratio"],
        default=scene_true_ratio,
    )
    style_match_rate = resolve_first_present(
        record,
        ["style_match_rate", "style_true_ratio"],
        default=style_true_ratio,
    )
    relation_match_rate = resolve_first_present(
        record,
        ["relation_match_rate", "relation_true_ratio"],
        default=relation_true_ratio,
    )
    count_match_rate = resolve_first_present(
        record,
        ["count_match_rate", "count_true_ratio"],
        default=count_true_ratio,
    )

    contradiction_default = 1.0 if (
        (subject_valid > 0.5 and subject_match < 0.5)
        or relation_false_ratio > 0.0
        or count_false_ratio > 0.0
    ) else 0.0
    contradiction_flag = 1.0 if resolve_first_present(
        record,
        ["contradiction_flag", "subject_hard_fail"],
        default=contradiction_default,
    ) >= 0.5 else 0.0

    logic = {
        "subject_match": subject_match,
        "present_ratio": present_ratio,
        "missing_ratio": missing_ratio,
        "off_topic_ratio": off_topic_ratio,
        "attribute_match_rate": attribute_match_rate,
        "scene_match_rate": scene_match_rate,
        "style_match_rate": style_match_rate,
        "relation_match_rate": relation_match_rate,
        "count_match_rate": count_match_rate,
        "confidence": confidence,
        "uncertainty_ratio": uncertainty_ratio,
        "contradiction_flag": contradiction_flag,
    }
    return logic


def logic_vector_from_record(record: Dict[str, Any]) -> List[float]:
    logic = logic_dict_from_record(record)
    return [logic[name] for name in FUNNEL_LOGIC_FEATURE_NAMES]


def compute_vlm_pseudo_label_from_funnel_record(record: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], float]:
    logic = logic_dict_from_record(record)
    subject_valid = has_value(record, "subject_match") or bool(safe_text(record.get("main_visual_subject")))
    coverage_valid = any(
        key in record for key in ("present_ratio", "missing_ratio", "uncertainty_ratio", "uncertain_ratio")
    )
    off_topic_valid = has_value(record, "off_topic_ratio") or has_value(record, "off_topic_concepts")
    attr_valid = any(has_value(record, key) for key in ("attribute_match_rate", "attr_true_ratio", "attribute_requirements", "attribute_match"))
    scene_valid = any(has_value(record, key) for key in ("scene_match_rate", "scene_true_ratio", "scene_requirements", "scene_match"))
    style_valid = any(has_value(record, key) for key in ("style_match_rate", "style_true_ratio", "style_requirements", "style_match"))
    relation_valid = any(has_value(record, key) for key in ("relation_match_rate", "relation_true_ratio", "relation_requirements", "relation_match"))
    count_valid = any(has_value(record, key) for key in ("count_match_rate", "count_true_ratio", "count_requirements", "count_match"))
    contradiction_valid = any(has_value(record, key) for key in ("contradiction_flag", "subject_match", "relation_requirements", "count_requirements"))
    components = [
        (logic["subject_match"], 0.38, subject_valid),
        (
            max(0.0, min(1.0, logic["present_ratio"] - 0.7 * logic["missing_ratio"] - 0.35 * logic["uncertainty_ratio"])),
            0.22,
            coverage_valid,
        ),
        (1.0 - logic["off_topic_ratio"], 0.10, off_topic_valid),
        (logic["attribute_match_rate"], 0.08, attr_valid),
        (logic["scene_match_rate"], 0.05, scene_valid),
        (logic["style_match_rate"], 0.04, style_valid),
        (logic["relation_match_rate"], 0.07, relation_valid),
        (logic["count_match_rate"], 0.03, count_valid),
        (1.0 - logic["contradiction_flag"], 0.03, contradiction_valid),
    ]
    total_weight = sum(weight for _, weight, valid in components if valid)
    if total_weight <= 0:
        return None, None, 0.0
    weighted_score = sum(score * weight for score, weight, valid in components if valid) / total_weight
    score = total_weight * weighted_score + (1.0 - total_weight) * 0.5
    score = clamp01(score)
    conf_default = max(0.3, min(1.0, total_weight))
    conf = clamp01(safe_float(record.get("confidence"), conf_default))
    return score, conf, 1.0


def merge_funnel_cache(
        base_df: pd.DataFrame,
        funnel_cache_jsonl: str,
        score_column: str = "vlm_align_score",
        conf_column: str = "vlm_confidence") -> pd.DataFrame:
    if not funnel_cache_jsonl:
        return base_df
    if not os.path.exists(funnel_cache_jsonl):
        raise FileNotFoundError(f"funnel_cache_jsonl not found: {funnel_cache_jsonl}")

    records: List[Dict[str, Any]] = []
    with open(funnel_cache_jsonl, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse funnel_cache_jsonl at line {line_idx}: {exc}") from exc
            if isinstance(obj, dict):
                records.append(obj)

    if not records:
        return base_df

    evidence_df = pd.DataFrame(records)
    if "name" in evidence_df.columns:
        evidence_df["name"] = evidence_df["name"].fillna("").astype(str).str.strip()
    if "prompt" in evidence_df.columns:
        evidence_df["prompt"] = evidence_df["prompt"].fillna("").astype(str).str.strip()

    for col in FUNNEL_CACHE_FIELDS:
        if col not in evidence_df.columns:
            if col in {"focus_bbox_xyxy"}:
                evidence_df[col] = None
            elif col.endswith("_path") or col in {"main_visual_subject"}:
                evidence_df[col] = ""
            else:
                evidence_df[col] = 0.0

    enriched = evidence_df.apply(lambda row: logic_dict_from_record(row.to_dict()), axis=1)
    for col in FUNNEL_LOGIC_FEATURE_NAMES:
        evidence_df[f"funnel_{col}"] = enriched.apply(lambda x, key=col: x[key])
    evidence_df["funnel_logic_vec"] = evidence_df.apply(lambda row: logic_vector_from_record(row.to_dict()), axis=1)
    evidence_df["funnel_bbox_feat"] = evidence_df.apply(lambda row: [], axis=1)
    pseudo = evidence_df.apply(lambda row: compute_vlm_pseudo_label_from_funnel_record(row.to_dict()), axis=1)
    evidence_df["__vlm_score_from_funnel"] = pseudo.apply(lambda x: x[0] if x[2] > 0 else np.nan)
    evidence_df["__vlm_conf_from_funnel"] = pseudo.apply(lambda x: x[1] if x[2] > 0 else np.nan)
    evidence_df["funnel_subject_text"] = evidence_df["main_visual_subject"].apply(safe_text)
    evidence_df["funnel_text_prompt"] = evidence_df["text_prompt"].apply(safe_text) if "text_prompt" in evidence_df.columns else ""
    evidence_df["funnel_selected_prompt"] = evidence_df["selected_prompt"].apply(safe_text) if "selected_prompt" in evidence_df.columns else ""
    evidence_df["funnel_selected_phrase"] = evidence_df["selected_phrase"].apply(safe_text) if "selected_phrase" in evidence_df.columns else ""
    evidence_df["funnel_focus_image_path"] = evidence_df["focus_image_path"].apply(safe_text)
    evidence_df["funnel_focus_mask_path"] = evidence_df["focus_mask_path"].apply(safe_text)

    df = base_df.copy()
    df["name"] = df["name"].astype(str).str.strip()
    df["prompt"] = df["prompt"].fillna("").astype(str).str.strip()

    added_row_index = False
    if "_row_index" not in df.columns:
        df["_row_index"] = np.arange(len(df))
        added_row_index = True

    if "_row_index" in evidence_df.columns and evidence_df["_row_index"].notna().all():
        merge_keys = ["_row_index"]
    elif "name" in evidence_df.columns and "prompt" in evidence_df.columns:
        merge_keys = ["name", "prompt"]
    elif "name" in evidence_df.columns:
        merge_keys = ["name"]
    else:
        raise ValueError("funnel_cache_jsonl must contain at least one of: _row_index, name, prompt.")

    evidence_df = evidence_df.drop_duplicates(subset=merge_keys, keep="first")
    merge_cols = merge_keys + FUNNEL_CACHE_FIELDS + [
        "funnel_logic_vec",
        "funnel_bbox_feat",
        "funnel_subject_text",
        "funnel_text_prompt",
        "funnel_selected_prompt",
        "funnel_selected_phrase",
        "funnel_focus_image_path",
        "funnel_focus_mask_path",
        "__vlm_score_from_funnel",
        "__vlm_conf_from_funnel",
    ] + [f"funnel_{name}" for name in FUNNEL_LOGIC_FEATURE_NAMES]
    df = df.merge(evidence_df[merge_cols], on=merge_keys, how="left")

    if score_column in df.columns:
        df[score_column] = df[score_column].where(df[score_column].notna(), df["__vlm_score_from_funnel"])
    else:
        df[score_column] = df["__vlm_score_from_funnel"]
    if conf_column in df.columns:
        df[conf_column] = df[conf_column].where(df[conf_column].notna(), df["__vlm_conf_from_funnel"])
    else:
        df[conf_column] = df["__vlm_conf_from_funnel"]
    df = df.drop(columns=["__vlm_score_from_funnel", "__vlm_conf_from_funnel"])
    if added_row_index:
        df = df.drop(columns=["_row_index"])
    return df


def _focus_mask_from_bbox(image_size: Tuple[int, int], bbox_xyxy: Optional[Tuple[float, float, float, float]]) -> Optional[Image.Image]:
    if bbox_xyxy is None:
        return None
    width, height = image_size
    x1, y1, x2, y2 = bbox_xyxy
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height
    x1 = int(max(0, min(width, round(x1))))
    y1 = int(max(0, min(height, round(y1))))
    x2 = int(max(0, min(width, round(x2))))
    y2 = int(max(0, min(height, round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    mask = Image.new("L", (width, height), color=0)
    for x in range(x1, x2):
        for y in range(y1, y2):
            mask.putpixel((x, y), 255)
    return mask


def bbox_from_mask(mask: Image.Image) -> Optional[Tuple[float, float, float, float]]:
    arr = np.asarray(mask, dtype=np.uint8)
    ys, xs = np.where(arr > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    width, height = mask.size
    x1 = float(xs.min()) / float(max(width, 1))
    y1 = float(ys.min()) / float(max(height, 1))
    x2 = float(xs.max() + 1) / float(max(width, 1))
    y2 = float(ys.max() + 1) / float(max(height, 1))
    return x1, y1, x2, y2


def render_focus_image(
        image_path: str,
        output_path: str,
        mask_path: str = "",
        bbox_xyxy: Optional[Sequence[float]] = None,
        blur_radius: float = 12.0) -> Dict[str, Any]:
    output = {
        "focus_image_path": output_path,
        "focus_mask_path": mask_path,
        "focus_valid": 0.0,
        "focus_bbox_xyxy": None,
        "focus_area_ratio": 0.0,
        "focus_center_x": 0.0,
        "focus_center_y": 0.0,
    }
    if not image_path or not os.path.exists(image_path):
        return output

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    mask_image: Optional[Image.Image] = None
    if mask_path and os.path.exists(mask_path):
        mask_image = Image.open(mask_path).convert("L").resize((width, height))
    else:
        mask_image = _focus_mask_from_bbox((width, height), tuple(bbox_xyxy) if bbox_xyxy is not None else None)

    if mask_image is None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        output["focus_image_path"] = output_path
        return output

    blurred = image.filter(ImageFilter.GaussianBlur(radius=float(max(0.0, blur_radius))))
    focus = Image.composite(image, blurred, mask_image)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    focus.save(output_path)

    bbox_xyxy_norm = bbox_from_mask(mask_image)
    features = bbox_features_from_xyxy(bbox_xyxy_norm)
    output.update({
        "focus_image_path": output_path,
        "focus_valid": 1.0,
        "focus_bbox_xyxy": list(bbox_xyxy_norm) if bbox_xyxy_norm is not None else None,
        "focus_area_ratio": features["bbox_area"],
        "focus_center_x": features["bbox_center_x"],
        "focus_center_y": features["bbox_center_y"],
    })
    return output


def load_jsonl_records(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _compute_ratio_from_list_fields(record: Dict[str, Any]) -> Dict[str, float]:
    present = parse_listish(record.get("present_requirements", record.get("present_concepts")))
    missing = parse_listish(record.get("missing_requirements", record.get("missing_concepts")))
    uncertain = parse_listish(record.get("uncertain_requirements", record.get("uncertain_concepts")))
    off_topic = parse_listish(record.get("off_topic_concepts"))
    total = float(max(len(present) + len(missing) + len(uncertain), 1))
    return {
        "present_ratio": clamp01(len(present) / total),
        "missing_ratio": clamp01(len(missing) / total),
        "uncertainty_ratio": clamp01(len(uncertain) / total),
        "off_topic_ratio": clamp01(len(off_topic) / float(max(len(off_topic) + int(total), 1))),
    }


def _mapping_true_false_ratio(value: Any) -> Tuple[float, float]:
    obj = parse_json_like(value, {})
    if isinstance(obj, dict):
        iterable = obj.values()
    elif isinstance(obj, (list, tuple)):
        iterable = obj
    else:
        return 0.0, 0.0
    true_cnt = 0.0
    false_cnt = 0.0
    for item in iterable:
        if isinstance(item, dict):
            item = item.get("match", item.get("status", item.get("value", item)))
        if isinstance(item, bool):
            true_cnt += float(item)
            false_cnt += float(not item)
        elif isinstance(item, (int, float)):
            true_cnt += float(item > 0)
            false_cnt += float(item <= 0)
        else:
            text = safe_text(item).lower()
            if text in {"true", "match", "matched", "yes", "present", "correct"}:
                true_cnt += 1.0
            elif text in {"false", "mismatch", "mismatched", "no", "missing", "wrong", "incorrect"}:
                false_cnt += 1.0
    total = max(true_cnt + false_cnt, 1.0)
    return true_cnt / total, false_cnt / total


def import_grounded_backend(spec: str) -> Callable[[str, Sequence[str]], Dict[str, Any]]:
    if ":" not in spec:
        raise ValueError("grounded backend must be in 'module:function' format.")
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    if not callable(func):
        raise TypeError(f"grounded backend '{spec}' is not callable.")
    return func


def build_funnel_cache_records(
        vlm_records: Iterable[Dict[str, Any]],
        output_focus_dir: str,
        image_base_dir: str = "",
        focus_asset_records: Optional[Iterable[Dict[str, Any]]] = None,
        grounded_backend: Optional[Callable[[str, Sequence[str]], Dict[str, Any]]] = None,
        blur_radius: float = 12.0) -> List[Dict[str, Any]]:
    focus_index: Dict[str, Dict[str, Any]] = {}
    if focus_asset_records is not None:
        for record in focus_asset_records:
            key = safe_text(record.get("_row_index"))
            if not key:
                key = f"{safe_text(record.get('name'))}||{safe_text(record.get('prompt'))}"
            focus_index[key] = dict(record)

    out_records: List[Dict[str, Any]] = []
    output_dir = Path(output_focus_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for row_idx, record in enumerate(vlm_records):
        row = dict(record)
        row_key = safe_text(row.get("_row_index"))
        if not row_key:
            row_key = f"{safe_text(row.get('name'))}||{safe_text(row.get('prompt'))}"
        image_path = safe_text(row.get("image_path"))
        if image_base_dir and image_path and not os.path.isabs(image_path):
            image_path = os.path.join(image_base_dir, image_path)

        main_subject = safe_text(row.get("main_visual_subject"))
        if "/" in main_subject:
            subject_prompts = [part.strip() for part in main_subject.split("/") if part.strip()]
        elif main_subject:
            subject_prompts = [main_subject]
        else:
            subject_prompts = []

        focus_payload = dict(focus_index.get(row_key, {}))
        if not focus_payload and grounded_backend is not None and image_path and subject_prompts:
            focus_payload = grounded_backend(image_path, subject_prompts) or {}

        logic = logic_dict_from_record(row)

        bbox_xyxy = parse_bbox_xyxy(
            focus_payload.get("focus_bbox_xyxy", focus_payload.get("bbox_xyxy", focus_payload.get("bbox")))
        )
        mask_path = safe_text(focus_payload.get("focus_mask_path", focus_payload.get("mask_path")))
        output_name = f"{safe_text(row.get('name')) or row_idx}.focus.png"
        output_path = str(output_dir / output_name)
        render_info = render_focus_image(
            image_path=image_path,
            output_path=output_path,
            mask_path=mask_path,
            bbox_xyxy=bbox_xyxy,
            blur_radius=blur_radius,
        )

        out = {
            "_row_index": row.get("_row_index"),
            "_row_key": row.get("_row_key"),
            "name": row.get("name"),
            "prompt": row.get("prompt"),
            "main_visual_subject": main_subject,
            "subject_prompts": focus_payload.get("subject_prompts", subject_prompts),
            "text_prompt": focus_payload.get("text_prompt", " . ".join(subject_prompts)),
            "selected_prompt": focus_payload.get("selected_prompt", ""),
            "selected_phrase": focus_payload.get("selected_phrase", ""),
            "detection_score": safe_float(focus_payload.get("detection_score"), 0.0),
            "sam_score": safe_float(focus_payload.get("sam_score"), 0.0),
            "focus_image_path": render_info["focus_image_path"],
            "focus_mask_path": mask_path,
            "focus_valid": render_info["focus_valid"],
            "focus_bbox_xyxy": render_info["focus_bbox_xyxy"],
            "focus_area_ratio": render_info["focus_area_ratio"],
            "focus_center_x": render_info["focus_center_x"],
            "focus_center_y": render_info["focus_center_y"],
            "error": safe_text(focus_payload.get("error")),
            **logic,
        }
        out_records.append(out)
    return out_records


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
