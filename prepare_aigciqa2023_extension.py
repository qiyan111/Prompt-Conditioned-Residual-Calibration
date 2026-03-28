#!/usr/bin/env python3
"""Prepare official AIGCIQA2023 annotations for the extended training pipeline.

This script converts the official split JSON files into:
1. A single CSV consumable by AIGCIQA2023_train.py / llm_prompt_parser.py
2. A split JSON consumable by --split_file

It also optionally joins generator metadata from AIGIQA2023+.csv so group-DRO can
use a meaningful generator column instead of falling back to filename prefixes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


COMBINED_QUESTION = "Assess the image from three perspectives: quality, authenticity, and correspondence"


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(out):
        return default
    return out


def load_json_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list.")
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"{path} item #{idx} is not a JSON object.")
        out.append(item)
    return out


def _normalize_question_key(question: str) -> str:
    text = safe_text(question).lower()
    if text == "what is the quality of the image?":
        return "quality"
    if text == "what is the authenticity of the image?":
        return "authenticity"
    if text == "what is the correspondence of the image?":
        return "correspondence"
    if COMBINED_QUESTION.lower() in text:
        return "combined"
    return "other"


def collapse_split_records(records: Sequence[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    ordered_ids: List[str] = []

    for row in records:
        sample_id = safe_text(row.get("id"))
        if not sample_id:
            raise ValueError(f"{split_name} split contains a row without 'id'.")
        if sample_id not in grouped:
            grouped[sample_id] = {
                "id": sample_id,
                "split": split_name,
                "json_name": safe_text(row.get("img")),
                "prompt": safe_text(row.get("prompt")),
                "moz1": safe_float(row.get("moz1")),
                "moz2": safe_float(row.get("moz2")),
                "moz3": safe_float(row.get("moz3")),
                "quality_explanation": "",
                "authenticity_explanation": "",
                "correspondence_explanation": "",
                "explanation": "",
            }
            ordered_ids.append(sample_id)

        entry = grouped[sample_id]
        question_key = _normalize_question_key(row.get("q1"))
        answer = safe_text(row.get("a1"))

        # Sanity-check duplicated scalar fields.
        for col in ("img", "prompt"):
            incoming = safe_text(row.get(col))
            existing = entry["json_name"] if col == "img" else entry["prompt"]
            if existing and incoming and existing != incoming:
                raise ValueError(
                    f"{split_name} split id={sample_id} has inconsistent {col}: "
                    f"{existing!r} vs {incoming!r}"
                )
        for col in ("moz1", "moz2", "moz3"):
            incoming = safe_float(row.get(col))
            existing = safe_float(entry[col])
            if abs(existing - incoming) > 1e-8:
                raise ValueError(
                    f"{split_name} split id={sample_id} has inconsistent {col}: "
                    f"{existing} vs {incoming}"
                )

        if question_key == "quality":
            entry["quality_explanation"] = answer
        elif question_key == "authenticity":
            entry["authenticity_explanation"] = answer
        elif question_key == "correspondence":
            entry["correspondence_explanation"] = answer
        elif question_key == "combined":
            entry["explanation"] = answer

    rows: List[Dict[str, Any]] = []
    for sample_id in ordered_ids:
        entry = grouped[sample_id]
        if not entry["explanation"]:
            parts = [
                entry["quality_explanation"],
                entry["authenticity_explanation"],
                entry["correspondence_explanation"],
            ]
            entry["explanation"] = " ".join(part for part in parts if part)
        rows.append(entry)
    return rows


def load_index_metadata(path: str, encoding: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, encoding=encoding)
    if df.shape[1] < 3:
        raise ValueError("AIGIQA2023+.csv must have at least 3 columns: id, filename, generator.")
    rename = {
        0: "id",
        1: "original_name",
        2: "generator",
    }
    df = df.rename(columns=rename)
    df["id"] = df["id"].astype(str).str.strip()
    df["original_name"] = df["original_name"].astype(str).str.strip()
    df["generator"] = df["generator"].fillna("").astype(str).str.strip()
    df = df.drop_duplicates(subset=["id"], keep="first")
    return df[["id", "original_name", "generator"]]


def choose_image_name(
    json_name: str,
    original_name: str,
    image_base_dir: str = "",
    prefer_original_names: bool = False,
) -> Tuple[str, str]:
    candidates: List[Tuple[str, str]] = []
    if prefer_original_names:
        candidates.extend([("original_name", original_name), ("json_name", json_name)])
    else:
        candidates.extend([("json_name", json_name), ("original_name", original_name)])

    cleaned = [(kind, safe_text(name)) for kind, name in candidates if safe_text(name)]
    if not cleaned:
        return "", ""

    if image_base_dir:
        for kind, name in cleaned:
            if os.path.exists(os.path.join(image_base_dir, name)):
                return name, kind

    return cleaned[0][1], cleaned[0][0]


def build_dataset_rows(
    train_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
    index_df: Optional[pd.DataFrame],
    image_base_dir: str = "",
    prefer_original_names: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    merged_rows = list(train_rows) + list(test_rows)
    df = pd.DataFrame(merged_rows)
    df["id"] = df["id"].astype(str).str.strip()

    if index_df is not None:
        df = df.merge(index_df, on="id", how="left")
    else:
        df["original_name"] = ""
        df["generator"] = ""

    chosen_names: List[str] = []
    chosen_name_sources: List[str] = []
    missing_name_count = 0

    for row in df.to_dict(orient="records"):
        chosen_name, source = choose_image_name(
            json_name=safe_text(row.get("json_name")),
            original_name=safe_text(row.get("original_name")),
            image_base_dir=image_base_dir,
            prefer_original_names=prefer_original_names,
        )
        chosen_names.append(chosen_name)
        chosen_name_sources.append(source)
        if not chosen_name:
            missing_name_count += 1

    df["name"] = chosen_names
    df["name_source"] = chosen_name_sources

    df["mos_quality"] = df["moz1"].apply(lambda x: safe_float(x) * 5.0)
    df["mos_authenticity"] = df["moz2"].apply(lambda x: safe_float(x) * 5.0)
    df["mos_align"] = df["moz3"].apply(lambda x: safe_float(x) * 5.0)

    output_df = df[
        [
            "id",
            "split",
            "name",
            "name_source",
            "json_name",
            "original_name",
            "generator",
            "prompt",
            "mos_quality",
            "mos_authenticity",
            "mos_align",
            "moz1",
            "moz2",
            "moz3",
            "explanation",
            "quality_explanation",
            "authenticity_explanation",
            "correspondence_explanation",
        ]
    ].copy()

    output_df = output_df.reset_index(drop=True)
    split_payload = {
        "format": "aigciqa2023_official_split_v1",
        "num_rows": int(len(output_df)),
        "train_ids": output_df.index[output_df["split"] == "train"].tolist(),
        "val_ids": output_df.index[output_df["split"] == "test"].tolist(),
        "stats": {
            "train_rows": int((output_df["split"] == "train").sum()),
            "val_rows": int((output_df["split"] == "test").sum()),
            "missing_name_rows": int(missing_name_count),
            "generator_available_rows": int((output_df["generator"].fillna("").astype(str).str.strip() != "").sum()),
            "name_source_counts": output_df["name_source"].value_counts(dropna=False).to_dict(),
        },
    }
    return output_df, split_payload


def main() -> None:
    parser = argparse.ArgumentParser("prepare_aigciqa2023_extension")
    parser.add_argument("--train_json", required=True, help="Path to mytraindict_llm_2023.json")
    parser.add_argument("--test_json", required=True, help="Path to mytestdict_llm_2023.json")
    parser.add_argument("--index_csv", default="", help="Optional path to AIGIQA2023+.csv")
    parser.add_argument("--index_csv_encoding", default="gb18030")
    parser.add_argument("--image_base_dir", default="", help="Optional image root for resolving actual filenames")
    parser.add_argument("--prefer_original_names", action="store_true",
                        help="Prefer original_name from AIGIQA2023+.csv over json_name when selecting `name`.")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_split_json", required=True)
    args = parser.parse_args()

    train_records = load_json_records(args.train_json)
    test_records = load_json_records(args.test_json)
    train_rows = collapse_split_records(train_records, split_name="train")
    test_rows = collapse_split_records(test_records, split_name="test")

    index_df: Optional[pd.DataFrame] = None
    if args.index_csv:
        index_df = load_index_metadata(args.index_csv, encoding=args.index_csv_encoding)

    output_df, split_payload = build_dataset_rows(
        train_rows=train_rows,
        test_rows=test_rows,
        index_df=index_df,
        image_base_dir=args.image_base_dir,
        prefer_original_names=args.prefer_original_names,
    )

    output_parent = Path(args.output_csv).resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_split_json).resolve().parent.mkdir(parents=True, exist_ok=True)

    output_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    with open(args.output_split_json, "w", encoding="utf-8") as f:
        json.dump(split_payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(
        {
            "output_csv": str(Path(args.output_csv).resolve()),
            "output_split_json": str(Path(args.output_split_json).resolve()),
            "rows": int(len(output_df)),
            "train_rows": split_payload["stats"]["train_rows"],
            "val_rows": split_payload["stats"]["val_rows"],
            "missing_name_rows": split_payload["stats"]["missing_name_rows"],
            "generator_available_rows": split_payload["stats"]["generator_available_rows"],
            "name_source_counts": split_payload["stats"]["name_source_counts"],
        },
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
