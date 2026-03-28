#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import io
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI
from PIL import Image
from tqdm import tqdm


SYSTEM_PROMPT = """
You are a strict image-text alignment auditor for text-to-image evaluation.

You will be given:
1. one image
2. one raw prompt

Your task is to jointly:
- identify the prompt requirements that matter for image-text consistency
- verify which requirements are satisfied by the image
- return structured JSON only

Do NOT judge aesthetics.
Do NOT explain your reasoning.
Do NOT return markdown.

Return exactly this JSON schema:
{
  "main_visual_subject": "",
  "subject_requirement": "",
  "subject_match": 0,
  "attribute_requirements": [{"text": "", "match": true}],
  "scene_requirements": [{"text": "", "match": true}],
  "style_requirements": [{"text": "", "match": true}],
  "relation_requirements": [{"text": "", "match": true}],
  "count_requirements": [{"text": "", "match": true}],
  "present_requirements": [],
  "missing_requirements": [],
  "uncertain_requirements": [],
  "off_topic_concepts": [],
  "confidence": 0.0
}

Rules:
1. Use short literal English phrases only.
2. Include only requirements that are explicitly expressed or strongly implied by the prompt.
3. For attribute/scene/style/relation/count requirements:
   - "match": true if clearly satisfied
   - "match": false if clearly violated or missing
   - "match": null if genuinely uncertain
4. Use subject_match = 1 only when the prompt's core subject clearly matches the image's main visual subject.
5. present_requirements / missing_requirements / uncertain_requirements should cover the main prompt requirements without overlap.
6. count_requirements should be used only for explicit numeric or quantified constraints in the prompt,
   such as: one/two/three, single, pair, several, multiple, crowd of, group of.
   Do NOT create count requirements from plain articles like "a" or "an".
7. off_topic_concepts should only contain major extra entities or scenes that are clearly visible and may hurt alignment.
   Do NOT include minor accessories, harmless clothing details, colors, rendering style, texture, quality, or camera terms.
8. Confidence must be conservative:
   - use 0.90-0.95 only for very clear cases
   - do not use 1.0 unless the case is essentially unambiguous
9. Be conservative. If uncertain, prefer null / uncertain rather than over-claiming.
""".strip()


COUNT_PATTERN = re.compile(
    r"\b("
    r"0|1|2|3|4|5|6|7|8|9|10|"
    r"one|two|three|four|five|six|seven|eight|nine|ten|"
    r"single|double|triple|pair|several|multiple|many|few|dozen|"
    r"group of|crowd of|swarm of|flock of|pack of|team of"
    r")\b",
    flags=re.IGNORECASE,
)

WEAK_OFF_TOPIC_TOKENS = {
    "3d",
    "render",
    "rendering",
    "appearance",
    "style",
    "styling",
    "detail",
    "details",
    "lighting",
    "light",
    "color",
    "colour",
    "texture",
    "textures",
    "quality",
    "resolution",
    "blur",
    "blurry",
    "background",
    "foreground",
    "collar",
    "shadow",
    "glow",
    "camera",
    "shot",
    "view",
}

RESAMPLE_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        text = safe_str(item)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def normalize_text(text: str) -> str:
    text = safe_str(text).lower()
    text = re.sub(r"[_/|]+", " ", text)
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(a|an|the)\s+", "", text)
    return text


def canonicalize_phrase(text: Any) -> str:
    raw = safe_str(text)
    if not raw:
        return ""
    raw = re.sub(r"\s+", " ", raw).strip()
    raw = re.sub(r"^(a|an|the)\s+", "", raw, flags=re.IGNORECASE)
    return raw.strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    text = safe_str(text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    text2 = re.sub(r"^`{3}(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text2 = re.sub(r"`{3}$", "", text2).strip()
    try:
        obj = json.loads(text2)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"Failed to parse JSON from model output: {text[:800]}")


def to_bool_or_none(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    text = safe_str(value).lower()
    if text in {"true", "1", "yes", "present", "match", "matched", "correct"}:
        return True
    if text in {"false", "0", "no", "missing", "mismatch", "wrong", "incorrect"}:
        return False
    if text in {"null", "none", "unknown", "uncertain", "unsure"}:
        return None
    return None


def sanitize_phrase_list(value: Any, limit: int = 24) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        items = [str(value)]
    out = dedup_keep_order([canonicalize_phrase(item) for item in items if canonicalize_phrase(item)])
    return out[:limit]


def sanitize_requirement_items(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        value = [value]

    out: List[Dict[str, Any]] = []
    seen = set()
    for item in value:
        if isinstance(item, dict):
            text = canonicalize_phrase(item.get("text") or item.get("requirement") or item.get("name"))
            match = to_bool_or_none(item.get("match"))
        else:
            text = canonicalize_phrase(item)
            match = None
        if not text:
            continue
        norm = normalize_text(text)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append({"text": text, "match": match})
    return out


def prompt_has_explicit_count(prompt: str) -> bool:
    return COUNT_PATTERN.search(safe_str(prompt)) is not None


def looks_like_count_requirement(text: str) -> bool:
    return COUNT_PATTERN.search(safe_str(text)) is not None


def filter_count_requirements(prompt: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if prompt_has_explicit_count(prompt):
        return items
    return [item for item in items if not looks_like_count_requirement(item["text"])]


def filter_count_like_phrases(prompt: str, phrases: List[str]) -> List[str]:
    if prompt_has_explicit_count(prompt):
        return phrases
    return [phrase for phrase in phrases if not looks_like_count_requirement(phrase)]


def sanitize_off_topic_concepts(value: Any, limit: int = 16) -> List[str]:
    phrases = sanitize_phrase_list(value, limit=limit * 2)
    out: List[str] = []
    for phrase in phrases:
        tokens = normalize_text(phrase).split()
        if not tokens:
            continue
        strong_tokens = [tok for tok in tokens if tok not in WEAK_OFF_TOPIC_TOKENS]
        if not strong_tokens:
            continue
        if any(tok in {"render", "appearance", "style", "lighting", "detail", "blur"} for tok in tokens) and len(strong_tokens) <= 1:
            continue
        out.append(phrase)
    return dedup_keep_order(out)[:limit]


def calibrate_confidence(
        value: Any,
        subject_match: int,
        missing_count: int,
        uncertain_count: int,
        off_topic_count: int,
        contradiction_flag: int) -> float:
    try:
        confidence = float(value)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(0.95, confidence))
    if contradiction_flag > 0 or subject_match == 0:
        confidence = min(confidence, 0.85)
    elif missing_count > 0:
        confidence = min(confidence, 0.90)
    elif uncertain_count > 0:
        confidence = min(confidence, 0.85)
    elif off_topic_count > 0:
        confidence = min(confidence, 0.90)
    return confidence


def resolve_requirement_conflicts(
        present: List[str],
        missing: List[str],
        uncertain: List[str]) -> Tuple[List[str], List[str], List[str]]:
    p = dedup_keep_order(present)
    m = dedup_keep_order(missing)
    u = dedup_keep_order(uncertain)
    p_set = {normalize_text(x) for x in p}
    m_set = {normalize_text(x) for x in m}
    u_set = {normalize_text(x) for x in u}
    conflicts = (p_set & m_set) | (p_set & u_set) | (m_set & u_set)
    if not conflicts:
        return p, m, u

    conflict_texts = []
    for src in (p, m, u):
        for item in src:
            if normalize_text(item) in conflicts:
                conflict_texts.append(item)
    u = dedup_keep_order(u + conflict_texts)
    p = [item for item in p if normalize_text(item) not in conflicts]
    m = [item for item in m if normalize_text(item) not in conflicts]
    return p, m, u


def rate_from_requirement_items(items: List[Dict[str, Any]]) -> Tuple[float, int, int, int]:
    if not items:
        return 0.0, 0, 0, 0
    true_count = sum(1 for item in items if item["match"] is True)
    false_count = sum(1 for item in items if item["match"] is False)
    known_count = true_count + false_count
    rate = float(true_count) / float(known_count) if known_count > 0 else 0.0
    return rate, true_count, false_count, known_count


def safe_ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def compute_derived_features(record: Dict[str, Any], prompt: str = "") -> Dict[str, Any]:
    subject_requirement = canonicalize_phrase(record.get("subject_requirement"))
    subject_match = 1 if int(record.get("subject_match", 0)) != 0 else 0

    attr_items = sanitize_requirement_items(record.get("attribute_requirements"))
    scene_items = sanitize_requirement_items(record.get("scene_requirements"))
    style_items = sanitize_requirement_items(record.get("style_requirements"))
    relation_items = sanitize_requirement_items(record.get("relation_requirements"))
    count_items = filter_count_requirements(prompt, sanitize_requirement_items(record.get("count_requirements")))

    present = filter_count_like_phrases(prompt, sanitize_phrase_list(record.get("present_requirements")))
    missing = filter_count_like_phrases(prompt, sanitize_phrase_list(record.get("missing_requirements")))
    uncertain = filter_count_like_phrases(prompt, sanitize_phrase_list(record.get("uncertain_requirements")))
    off_topic = sanitize_off_topic_concepts(record.get("off_topic_concepts"), limit=16)

    status_map: Dict[str, str] = {}

    def _put_status(text: str, status: str) -> None:
        norm = normalize_text(text)
        if not norm:
            return
        prev = status_map.get(norm)
        if prev is None:
            status_map[norm] = status
        elif prev != status:
            status_map[norm] = "uncertain"

    if subject_requirement:
        _put_status(subject_requirement, "present" if subject_match else "missing")

    for item in attr_items + scene_items + style_items + relation_items + count_items:
        if item["match"] is True:
            _put_status(item["text"], "present")
        elif item["match"] is False:
            _put_status(item["text"], "missing")
        else:
            _put_status(item["text"], "uncertain")

    for text in present:
        _put_status(text, "present")
    for text in missing:
        _put_status(text, "missing")
    for text in uncertain:
        _put_status(text, "uncertain")

    present_merged = [text for text, status in ((subject_requirement, "present" if subject_match else "missing"),) if text and status_map.get(normalize_text(text)) == "present"]
    missing_merged = [text for text, status in ((subject_requirement, "present" if subject_match else "missing"),) if text and status_map.get(normalize_text(text)) == "missing"]
    uncertain_merged = [text for text in [subject_requirement] if text and status_map.get(normalize_text(text)) == "uncertain"]

    for collection, target in (
            (attr_items + scene_items + style_items + relation_items + count_items, (present_merged, missing_merged, uncertain_merged)),
            ([{"text": t, "match": True} for t in present], (present_merged, missing_merged, uncertain_merged)),
            ([{"text": t, "match": False} for t in missing], (present_merged, missing_merged, uncertain_merged)),
            ([{"text": t, "match": None} for t in uncertain], (present_merged, missing_merged, uncertain_merged))):
        for item in collection:
            text = safe_str(item["text"])
            norm = normalize_text(text)
            if not norm:
                continue
            status = status_map.get(norm)
            if status == "present":
                target[0].append(text)
            elif status == "missing":
                target[1].append(text)
            elif status == "uncertain":
                target[2].append(text)

    present_merged, missing_merged, uncertain_merged = resolve_requirement_conflicts(
        present_merged, missing_merged, uncertain_merged
    )

    total_requirements = len({
        normalize_text(text)
        for text in present_merged + missing_merged + uncertain_merged
        if normalize_text(text)
    })

    attr_rate, attr_true, attr_false, attr_known = rate_from_requirement_items(attr_items)
    scene_rate, scene_true, scene_false, scene_known = rate_from_requirement_items(scene_items)
    style_rate, style_true, style_false, style_known = rate_from_requirement_items(style_items)
    relation_rate, relation_true, relation_false, relation_known = rate_from_requirement_items(relation_items)
    count_rate, count_true, count_false, count_known = rate_from_requirement_items(count_items)

    contradiction_flag = 1 if (
        (subject_requirement and subject_match == 0)
        or relation_false > 0
        or count_false > 0
    ) else 0

    confidence = calibrate_confidence(
        record.get("confidence", 0.0),
        subject_match=subject_match,
        missing_count=len(missing_merged),
        uncertain_count=len(uncertain_merged),
        off_topic_count=len(off_topic),
        contradiction_flag=contradiction_flag,
    )

    return {
        "subject_requirement": subject_requirement,
        "subject_match": subject_match,
        "attribute_requirements": attr_items,
        "scene_requirements": scene_items,
        "style_requirements": style_items,
        "relation_requirements": relation_items,
        "count_requirements": count_items,
        "present_requirements": present_merged,
        "missing_requirements": missing_merged,
        "uncertain_requirements": uncertain_merged,
        "off_topic_concepts": off_topic,
        "total_requirement_count": total_requirements,
        "present_requirement_count": len(present_merged),
        "missing_requirement_count": len(missing_merged),
        "uncertain_requirement_count": len(uncertain_merged),
        "off_topic_count": len(off_topic),
        "present_ratio": safe_ratio(len(present_merged), total_requirements),
        "missing_ratio": safe_ratio(len(missing_merged), total_requirements),
        "uncertainty_ratio": safe_ratio(len(uncertain_merged), total_requirements),
        "off_topic_ratio": safe_ratio(len(off_topic), total_requirements + len(off_topic)),
        "attribute_match_rate": attr_rate,
        "scene_match_rate": scene_rate,
        "style_match_rate": style_rate,
        "relation_match_rate": relation_rate,
        "count_match_rate": count_rate,
        "attribute_known_count": attr_known,
        "scene_known_count": scene_known,
        "style_known_count": style_known,
        "relation_known_count": relation_known,
        "count_known_count": count_known,
        "contradiction_flag": contradiction_flag,
        "confidence": confidence,
    }


def guess_mime_type(path: str) -> str:
    ext = os.path.splitext(path.lower())[1]
    if ext == ".png":
        return "image/png"
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"


def format_name_from_mime(mime: str) -> str:
    if mime == "image/png":
        return "PNG"
    if mime == "image/webp":
        return "WEBP"
    return "JPEG"


def image_to_data_url(image_path: str, max_image_side: int = 1024, jpeg_quality: int = 90) -> str:
    mime = guess_mime_type(image_path)
    output_mime = mime if mime in {"image/png", "image/jpeg", "image/webp"} else "image/jpeg"
    output_format = format_name_from_mime(output_mime)

    with Image.open(image_path) as image:
        image.load()
        width, height = image.size
        if max_image_side > 0 and max(width, height) > max_image_side:
            scale = float(max_image_side) / float(max(width, height))
            new_size = (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )
            image = image.resize(new_size, RESAMPLE_LANCZOS)

        if output_format in {"JPEG", "WEBP"} and image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")

        buffer = io.BytesIO()
        save_kwargs: Dict[str, Any] = {}
        if output_format == "JPEG":
            save_kwargs = {"quality": jpeg_quality, "optimize": True}
        elif output_format == "WEBP":
            save_kwargs = {"quality": jpeg_quality, "method": 4}
        elif output_format == "PNG":
            save_kwargs = {"optimize": True}
        image.save(buffer, format=output_format, **save_kwargs)

    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = output_mime
    return f"data:{mime};base64,{b64}"


def make_row_key(idx: int, row: pd.Series) -> str:
    return f'{idx}||{safe_str(row.get("name", ""))}||{safe_str(row.get("prompt", ""))}'


def find_image_path(image_root: str, row: pd.Series) -> Optional[str]:
    raw_image_path = safe_str(row.get("image_path"))
    if raw_image_path and os.path.exists(raw_image_path):
        return raw_image_path

    name = safe_str(row.get("name"))
    if image_root and name:
        candidate = os.path.join(image_root, name)
        if os.path.exists(candidate):
            return candidate
        stem, _ = os.path.splitext(name)
        for ext in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"]:
            candidate = os.path.join(image_root, stem + ext)
            if os.path.exists(candidate):
                return candidate
    return None


def load_done(output_jsonl: str) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    if not output_jsonl or not os.path.exists(output_jsonl):
        return done
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            key = safe_str(obj.get("_row_key"))
            if not key:
                continue
            error_msg = safe_str(obj.get("_error"))
            if error_msg and error_msg != "image_not_found":
                if key in done:
                    del done[key]
            else:
                done[key] = obj
    return done


class DirectAlignmentVerifier:
    def __init__(
            self,
            model: str,
            api_key: str,
            base_url: Optional[str] = None,
            temperature: float = 0.0,
            max_retries: int = 5,
            timeout: float = 600.0,
            max_image_side: int = 1024):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_image_side = max_image_side
        self._local = threading.local()

    def _get_client(self) -> OpenAI:
        client = getattr(self._local, "client", None)
        if client is None:
            kwargs = {"api_key": self.api_key, "timeout": self.timeout}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            client = OpenAI(**kwargs)
            self._local.client = client
        return client

    def verify_one(self, image_path: str, prompt: str) -> Dict[str, Any]:
        started_at = time.perf_counter()
        data_url = image_to_data_url(image_path, max_image_side=self.max_image_side)
        user_text = json.dumps(
            {
                "task": "Audit image-text alignment and return structured requirement-level JSON.",
                "prompt": prompt,
            },
            ensure_ascii=False,
            indent=2,
        )

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                client = self._get_client()
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
                if not resp.choices or not resp.choices[0].message.content:
                    raise ValueError("Empty response received from the model.")
                obj = extract_json_object(resp.choices[0].message.content)
                elapsed_s = time.perf_counter() - started_at
                return obj, elapsed_s
            except Exception as exc:
                last_err = exc
                if attempt < self.max_retries:
                    time.sleep(5 * attempt)
                else:
                    raise last_err


def flatten_record(record: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, list):
            flat[key] = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, dict):
            flat[key] = json.dumps(value, ensure_ascii=False)
        else:
            flat[key] = value
    return flat


def as_completed_wrapper(tasks):
    future_to_payload = {fut: payload for payload, fut in tasks}
    for fut in as_completed(future_to_payload):
        yield future_to_payload[fut], fut


def process_all(
        input_csv: str,
        image_root: str,
        output_jsonl: str,
        output_csv: str,
        model: str,
        api_key: str,
        base_url: str,
        max_workers: int,
        limit: int = 0,
        flush_every: int = 10,
        max_image_side: int = 1024) -> None:
    df = pd.read_csv(input_csv)
    verifier = DirectAlignmentVerifier(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        max_image_side=max_image_side,
    )
    done = load_done(output_jsonl)
    records: List[Dict[str, Any]] = []
    tasks = []
    file_lock = threading.Lock()
    flush_every = max(1, int(flush_every))
    overall_started_at = time.perf_counter()

    print(f"Skipping {len(done)} already successfully processed records...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        processed_count = 0
        for idx, row in df.iterrows():
            if limit > 0 and processed_count >= limit:
                break
            processed_count += 1

            row_key = make_row_key(idx, row)
            if row_key in done:
                records.append(done[row_key])
                continue

            image_path = find_image_path(image_root, row)
            if image_path is None:
                rec = {
                    "_row_index": idx,
                    "_row_key": row_key,
                    "name": safe_str(row.get("name")),
                    "prompt": safe_str(row.get("prompt")),
                    "image_path": "",
                    "_error": "image_not_found",
                }
                records.append(rec)
                continue

            fut = executor.submit(verifier.verify_one, image_path, safe_str(row.get("prompt")))
            tasks.append(((idx, row, row_key, image_path), fut))

        if tasks:
            with open(output_jsonl, "a", encoding="utf-8") as fout:
                pending_flush_count = 0
                completed_count = 0
                timed_success_count = 0
                overall_elapsed_sum = 0.0
                batch_elapsed_sum = 0.0
                batch_timed_count = 0
                batch_index = 0
                progress = tqdm(
                    as_completed_wrapper(tasks),
                    total=len(tasks),
                    desc="Direct VLM Alignment",
                )
                for (idx, row, row_key, image_path), fut in progress:
                    sample_elapsed_s: Optional[float] = None
                    try:
                        raw, sample_elapsed_s = fut.result()
                        normalized = {
                            "main_visual_subject": safe_str(raw.get("main_visual_subject")),
                            "subject_requirement": canonicalize_phrase(raw.get("subject_requirement")),
                            "subject_match": 1 if int(raw.get("subject_match", 0)) != 0 else 0,
                            "attribute_requirements": sanitize_requirement_items(raw.get("attribute_requirements")),
                            "scene_requirements": sanitize_requirement_items(raw.get("scene_requirements")),
                            "style_requirements": sanitize_requirement_items(raw.get("style_requirements")),
                            "relation_requirements": sanitize_requirement_items(raw.get("relation_requirements")),
                            "count_requirements": sanitize_requirement_items(raw.get("count_requirements")),
                            "present_requirements": sanitize_phrase_list(raw.get("present_requirements")),
                            "missing_requirements": sanitize_phrase_list(raw.get("missing_requirements")),
                            "uncertain_requirements": sanitize_phrase_list(raw.get("uncertain_requirements")),
                            "off_topic_concepts": sanitize_off_topic_concepts(raw.get("off_topic_concepts"), limit=16),
                            "confidence": raw.get("confidence", 0.0),
                        }
                        derived = compute_derived_features(normalized, prompt=safe_str(row.get("prompt")))
                        rec = {
                            "_row_index": idx,
                            "_row_key": row_key,
                            "name": safe_str(row.get("name")),
                            "prompt": safe_str(row.get("prompt")),
                            "image_path": image_path,
                            "main_visual_subject": normalized["main_visual_subject"],
                            **derived,
                        }
                    except Exception as exc:
                        rec = {
                            "_row_index": idx,
                            "_row_key": row_key,
                            "name": safe_str(row.get("name")),
                            "prompt": safe_str(row.get("prompt")),
                            "image_path": image_path,
                            "_error": safe_str(exc),
                        }

                    with file_lock:
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        pending_flush_count += 1
                        if pending_flush_count >= flush_every:
                            fout.flush()
                            batch_index += 1
                            batch_avg_s = (batch_elapsed_sum / batch_timed_count) if batch_timed_count > 0 else 0.0
                            overall_avg_s = (overall_elapsed_sum / timed_success_count) if timed_success_count > 0 else 0.0
                            print(
                                f"[flush] batch={batch_index} size={pending_flush_count} "
                                f"written={completed_count + 1} batch_avg_s={batch_avg_s:.2f} "
                                f"overall_avg_s={overall_avg_s:.2f}"
                            )
                            pending_flush_count = 0
                            batch_elapsed_sum = 0.0
                            batch_timed_count = 0
                    records.append(rec)

                    completed_count += 1
                    if sample_elapsed_s is not None:
                        timed_success_count += 1
                        overall_elapsed_sum += sample_elapsed_s
                        batch_elapsed_sum += sample_elapsed_s
                        batch_timed_count += 1
                        overall_avg_s = overall_elapsed_sum / timed_success_count
                        progress.set_postfix(
                            last_s=f"{sample_elapsed_s:.2f}",
                            avg_s=f"{overall_avg_s:.2f}",
                            flush_n=pending_flush_count,
                        )

                if pending_flush_count > 0:
                    fout.flush()
                    batch_index += 1
                    batch_avg_s = (batch_elapsed_sum / batch_timed_count) if batch_timed_count > 0 else 0.0
                    overall_avg_s = (overall_elapsed_sum / timed_success_count) if timed_success_count > 0 else 0.0
                    print(
                        f"[flush] batch={batch_index} size={pending_flush_count} "
                        f"written={completed_count} batch_avg_s={batch_avg_s:.2f} "
                        f"overall_avg_s={overall_avg_s:.2f}"
                    )

    wall_elapsed_s = time.perf_counter() - overall_started_at
    timed_avg_s = 0.0
    timed_count = sum(1 for record in records if "_error" not in record)
    if timed_count > 0:
        # `records` may include skipped rows and image-not-found rows, so re-estimate with successful processed count.
        timed_avg_s = wall_elapsed_s / max(1, timed_count)
    print(
        f"[timing] wall_s={wall_elapsed_s:.2f} total_records={len(records)} "
        f"successful_records={timed_count} wall_avg_s={timed_avg_s:.2f}"
    )

    merged = {}
    for key, value in done.items():
        merged[key] = value
    for record in records:
        merged[safe_str(record.get("_row_key"))] = record

    merged_list = sorted(merged.values(), key=lambda item: int(item.get("_row_index", 0)))
    out_df = pd.DataFrame([flatten_record(record) for record in merged_list])
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser("verify_images_direct_alignment_features")
    parser.add_argument("--input_csv", required=True, type=str)
    parser.add_argument("--image_root", required=True, type=str)
    parser.add_argument("--output_jsonl", required=True, type=str)
    parser.add_argument("--output_csv", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--api_key", default="llama", type=str)
    parser.add_argument("--base_url", default="http://127.0.0.1:8080/v1", type=str)
    parser.add_argument("--max_workers", default=2, type=int)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--flush_every", default=10, type=int)
    parser.add_argument("--max_image_side", default=1024, type=int)
    args = parser.parse_args()

    process_all(
        input_csv=args.input_csv,
        image_root=args.image_root,
        output_jsonl=args.output_jsonl,
        output_csv=args.output_csv,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        max_workers=args.max_workers,
        limit=args.limit,
        flush_every=args.flush_every,
        max_image_side=args.max_image_side,
    )
    print(f"[OK] saved jsonl to {args.output_jsonl}")
    print(f"[OK] saved csv to {args.output_csv}")


if __name__ == "__main__":
    main()
