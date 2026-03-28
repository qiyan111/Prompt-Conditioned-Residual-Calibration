#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


ALIGNMENT_LOGIC_12D_NAMES = [
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

FIELD_ALIASES = {
    "attribute_match_rate": ["attribute_match_rate", "attr_true_ratio"],
    "scene_match_rate": ["scene_match_rate", "scene_true_ratio"],
    "style_match_rate": ["style_match_rate", "style_true_ratio"],
    "uncertainty_ratio": ["uncertainty_ratio", "uncertain_ratio"],
}

ID_FIELDS = ["_row_index", "_row_key", "name", "prompt", "image_path"]


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def clamp01(value: Any, default: float = 0.0) -> float:
    return float(max(0.0, min(1.0, safe_float(value, default))))


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except Exception as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} is not a JSON object.")
            records.append(obj)
    return records


def split_valid_records(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    valid: List[Dict[str, Any]] = []
    invalid: List[Dict[str, Any]] = []
    for record in records:
        if record.get("_error"):
            invalid.append(record)
        else:
            valid.append(record)
    return valid, invalid


def resolve_feature_value(record: Dict[str, Any], name: str) -> float:
    candidate_names = FIELD_ALIASES.get(name, [name])
    for candidate in candidate_names:
        if candidate in record:
            return clamp01(record.get(candidate), 0.0)
    return 0.0


def build_vector_record(record: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {field: record.get(field, "") for field in ID_FIELDS}
    feature_values: Dict[str, float] = {}
    for name in ALIGNMENT_LOGIC_12D_NAMES:
        feature_values[name] = resolve_feature_value(record, name)

    vector = [feature_values[name] for name in ALIGNMENT_LOGIC_12D_NAMES]
    out.update(feature_values)
    out["alignment_logic_dim"] = len(ALIGNMENT_LOGIC_12D_NAMES)
    out["alignment_logic_feature_names"] = list(ALIGNMENT_LOGIC_12D_NAMES)
    out["alignment_logic_vec_12d"] = vector
    return out


def flatten_for_csv(record: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (list, dict)):
            flat[key] = json.dumps(value, ensure_ascii=False)
        else:
            flat[key] = value
    return flat


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser("vectorize_direct_alignment_features")
    parser.add_argument("--input_jsonl", required=True, type=str)
    parser.add_argument("--output_jsonl", required=True, type=str)
    parser.add_argument("--output_csv", required=True, type=str)
    args = parser.parse_args()

    src_records = read_jsonl(args.input_jsonl)
    valid_records, invalid_records = split_valid_records(src_records)
    out_records = [build_vector_record(record) for record in valid_records]

    write_jsonl(args.output_jsonl, out_records)
    pd.DataFrame([flatten_for_csv(record) for record in out_records]).to_csv(
        args.output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[OK] read {len(src_records)} records from {args.input_jsonl}")
    print(f"[OK] vectorized {len(valid_records)} valid records")
    if invalid_records:
        print(f"[WARN] skipped {len(invalid_records)} records with _error")
    print(f"[OK] saved vector jsonl to {args.output_jsonl}")
    print(f"[OK] saved vector csv to {args.output_csv}")
    print(f"[OK] feature order: {ALIGNMENT_LOGIC_12D_NAMES}")


if __name__ == "__main__":
    main()
