#!/usr/bin/env python3
"""Model-specific adapter that runs inside the target conda environment."""

import argparse
import copy
import csv
import glob
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import types
from itertools import product
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import shlex

import numpy as np
try:
    import pandas as pd
except ModuleNotFoundError:
    # Some external model envs do not ship pandas; the adapter only needs a
    # small subset of DataFrame/Series operations, so keep a local fallback.
    def _compat_is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value == ""
        return isinstance(value, (float, np.floating)) and math.isnan(float(value))


    class _CompatSeries:
        def __init__(self, values: Sequence[Any]) -> None:
            self._values = list(values)

        def astype(self, dtype: Any) -> "_CompatSeries":
            if dtype in (str, "str", "string"):
                return _CompatSeries(["" if _compat_is_missing(value) else str(value) for value in self._values])
            if dtype in (float, "float", "float64"):
                return _CompatSeries([float(value) for value in self._values])
            if dtype in (int, "int", "int64"):
                return _CompatSeries([int(float(value)) for value in self._values])
            return _CompatSeries([dtype(value) for value in self._values])

        def fillna(self, value: Any) -> "_CompatSeries":
            return _CompatSeries([value if _compat_is_missing(item) else item for item in self._values])

        def tolist(self) -> List[Any]:
            return list(self._values)

        def __iter__(self):
            return iter(self._values)

        def __len__(self) -> int:
            return len(self._values)


    class _CompatILoc:
        def __init__(self, frame: "_CompatDataFrame") -> None:
            self._frame = frame

        def __getitem__(self, key: Any) -> Any:
            if isinstance(key, slice):
                return _CompatDataFrame(self._frame._rows[key], columns=self._frame._columns)
            return copy.deepcopy(self._frame._rows[key])


    class _CompatDataFrame:
        def __init__(self, data: Optional[Any] = None, columns: Optional[Sequence[str]] = None) -> None:
            self._columns = list(columns) if columns is not None else []
            rows: List[Dict[str, Any]] = []
            if data is None:
                self._rows = rows
                return

            if isinstance(data, dict):
                if not self._columns:
                    self._columns = [str(key) for key in data.keys()]
                normalized: Dict[str, List[Any]] = {
                    str(col): list(data.get(col, []))
                    for col in self._columns
                }
                row_count = max((len(values) for values in normalized.values()), default=0)
                for idx in range(row_count):
                    row = {}
                    for col in self._columns:
                        values = normalized[col]
                        row[col] = values[idx] if idx < len(values) else ""
                    rows.append(row)
                self._rows = rows
                return

            raw_rows = [dict(item) for item in data]
            if not self._columns:
                seen = set()
                for row in raw_rows:
                    for col in row.keys():
                        col_str = str(col)
                        if col_str not in seen:
                            seen.add(col_str)
                            self._columns.append(col_str)
            self._rows = [
                {col: row.get(col, "") for col in self._columns}
                for row in raw_rows
            ]

        @property
        def empty(self) -> bool:
            return len(self._rows) == 0

        @property
        def iloc(self) -> _CompatILoc:
            return _CompatILoc(self)

        @property
        def columns(self) -> List[str]:
            return list(self._columns)

        def __len__(self) -> int:
            return len(self._rows)

        def __getitem__(self, key: Any) -> Any:
            if isinstance(key, str):
                return _CompatSeries([row.get(key, "") for row in self._rows])
            if isinstance(key, (list, tuple)):
                selected = [str(col) for col in key]
                return _CompatDataFrame(
                    [{col: row.get(col, "") for col in selected} for row in self._rows],
                    columns=selected,
                )
            raise TypeError(f"Unsupported key type for DataFrame access: {type(key)!r}")

        def __setitem__(self, key: str, value: Any) -> None:
            key = str(key)
            if key not in self._columns:
                self._columns.append(key)

            if isinstance(value, _CompatSeries):
                values = value.tolist()
                scalar_value = None
            elif isinstance(value, (list, tuple)):
                values = list(value)
                scalar_value = None
            else:
                values = []
                scalar_value = value

            if not self._rows and scalar_value is None:
                self._rows = [{col: "" for col in self._columns} for _ in range(len(values))]

            if scalar_value is None and len(values) != len(self._rows):
                raise ValueError("Assigned column length does not match DataFrame length.")

            for idx, row in enumerate(self._rows):
                row[key] = scalar_value if scalar_value is not None else values[idx]

        def copy(self) -> "_CompatDataFrame":
            return _CompatDataFrame(copy.deepcopy(self._rows), columns=self._columns)

        def reset_index(self, drop: bool = False) -> "_CompatDataFrame":
            if not drop:
                raise NotImplementedError("Compat DataFrame only supports reset_index(drop=True).")
            return self.copy()

        def to_csv(self, path: str, index: bool = False, encoding: str = "utf-8") -> None:
            if index:
                raise NotImplementedError("Compat DataFrame does not support index=True in to_csv.")
            with open(path, "w", encoding=encoding, newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._columns)
                writer.writeheader()
                for row in self._rows:
                    writer.writerow({col: row.get(col, "") for col in self._columns})


    def _compat_read_csv(path: str) -> _CompatDataFrame:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = [dict(row) for row in reader]
            return _CompatDataFrame(rows, columns=reader.fieldnames or [])


    def _compat_concat(frames: Sequence[_CompatDataFrame], ignore_index: bool = False) -> _CompatDataFrame:
        if not ignore_index:
            raise NotImplementedError("Compat pandas only supports ignore_index=True in concat.")
        merged_columns: List[str] = []
        seen = set()
        merged_rows: List[Dict[str, Any]] = []
        for frame in frames:
            for col in frame.columns:
                if col not in seen:
                    seen.add(col)
                    merged_columns.append(col)
            merged_rows.extend(copy.deepcopy(frame._rows))
        return _CompatDataFrame(merged_rows, columns=merged_columns)


    def _compat_to_numeric(values: Any, errors: str = "raise") -> _CompatSeries:
        if isinstance(values, _CompatSeries):
            raw_values = values.tolist()
        else:
            raw_values = list(values)
        numeric: List[Any] = []
        for value in raw_values:
            try:
                if _compat_is_missing(value):
                    raise ValueError("missing")
                numeric.append(float(value))
            except (TypeError, ValueError):
                if errors == "coerce":
                    numeric.append(float("nan"))
                else:
                    raise
        return _CompatSeries(numeric)


    class _CompatPandasModule:
        DataFrame = _CompatDataFrame
        Series = _CompatSeries

        @staticmethod
        def read_csv(path: str) -> _CompatDataFrame:
            return _compat_read_csv(path)

        @staticmethod
        def concat(frames: Sequence[_CompatDataFrame], ignore_index: bool = False) -> _CompatDataFrame:
            return _compat_concat(frames, ignore_index=ignore_index)

        @staticmethod
        def to_numeric(values: Any, errors: str = "raise") -> _CompatSeries:
            return _compat_to_numeric(values, errors=errors)


    pd = _CompatPandasModule()


def _repair_broken_compiler_env() -> None:
    fallback_by_key = {
        "CC": shutil.which("gcc") or "",
        "CXX": shutil.which("g++") or "",
        "CUDAHOSTCXX": shutil.which("g++") or "",
        "CPP": shutil.which("cpp") or "",
    }
    suspicious_names = {
        "x86_64-conda-linux-gnu-gcc",
        "x86_64-conda-linux-gnu-g++",
    }

    def _sanitize_command_var(key: str) -> None:
        value = os.environ.get(key, "").strip()
        if not value:
            return
        parts = shlex.split(value)
        if not parts:
            return
        exe = parts[0]
        exe_name = os.path.basename(exe)
        exe_missing = os.path.isabs(exe) and not os.path.exists(exe)
        exe_suspicious = exe_name in suspicious_names and shutil.which(exe_name) is None and not os.path.exists(exe)
        if not (exe_missing or exe_suspicious):
            return
        fallback = fallback_by_key.get(key, "")
        if fallback:
            parts[0] = fallback
            os.environ[key] = " ".join(parts)
        else:
            os.environ.pop(key, None)

    for env_key in ("CC", "CXX", "CUDAHOSTCXX", "CPP"):
        _sanitize_command_var(env_key)

    ldshared = os.environ.get("LDSHARED", "").strip()
    if ldshared:
        parts = shlex.split(ldshared)
        if parts:
            exe = parts[0]
            exe_name = os.path.basename(exe)
            exe_missing = os.path.isabs(exe) and not os.path.exists(exe)
            exe_suspicious = exe_name in suspicious_names and shutil.which(exe_name) is None and not os.path.exists(exe)
            if exe_missing or exe_suspicious:
                fallback = fallback_by_key["CC"]
                if fallback:
                    parts[0] = fallback
                    os.environ["LDSHARED"] = " ".join(parts)
                else:
                    os.environ.pop("LDSHARED", None)


_repair_broken_compiler_env()

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
try:
    from scipy.stats import spearmanr as scipy_spearmanr
except ModuleNotFoundError:
    scipy_spearmanr = None
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True
QUALITY_PROMPT_WORDS = ["badly", "poorly", "fairly", "well", "perfectly"]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return int(sum(p.numel() for p in params))


def _average_tie_ranks(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    order = np.argsort(arr, kind="mergesort")
    sorted_arr = arr[order]
    ranks = np.empty(arr.shape[0], dtype=np.float64)
    start = 0
    while start < sorted_arr.shape[0]:
        end = start + 1
        while end < sorted_arr.shape[0] and sorted_arr[end] == sorted_arr[start]:
            end += 1
        avg_rank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _numpy_spearmanr(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred_arr = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    valid_mask = np.isfinite(true_arr) & np.isfinite(pred_arr)
    true_arr = true_arr[valid_mask]
    pred_arr = pred_arr[valid_mask]
    if true_arr.shape[0] < 2 or pred_arr.shape[0] < 2:
        return float("nan")

    rank_true = _average_tie_ranks(true_arr)
    rank_pred = _average_tie_ranks(pred_arr)
    if np.all(rank_true == rank_true[0]) or np.all(rank_pred == rank_pred[0]):
        return float("nan")

    corr = np.corrcoef(rank_true, rank_pred)[0, 1]
    return float(corr)


def safe_srocc(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.0
    if scipy_spearmanr is not None:
        value = scipy_spearmanr(np.asarray(y_true), np.asarray(y_pred)).correlation
    else:
        value = _numpy_spearmanr(y_true, y_pred)
    if value is None or not np.isfinite(value):
        return 0.0
    return float(value)


def save_prediction_csv(path: str, names: Sequence[str], preds: Sequence[float]) -> None:
    pd.DataFrame({"name": list(names), "pred_raw": list(preds)}).to_csv(path, index=False, encoding="utf-8")


def snapshot_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_snapshot(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state_dict, strict=True)


def build_interpolation() -> Any:
    if hasattr(transforms, "InterpolationMode"):
        return transforms.InterpolationMode.BICUBIC
    return Image.BICUBIC


class AdaptiveResize:
    def __init__(self, size: int, interpolation: Any) -> None:
        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if min(w, h) >= self.size:
            return img
        return transforms.Resize(self.size, interpolation=self.interpolation)(img)


def build_ipce_transform(train: bool) -> transforms.Compose:
    interpolation = build_interpolation()
    ops: List[Any] = [lambda image: image.convert("RGB"), AdaptiveResize(512, interpolation)]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    return transforms.Compose(ops)


def build_clip_agiqa_transform(train: bool) -> transforms.Compose:
    interpolation = build_interpolation()
    if train:
        crop = transforms.RandomResizedCrop(224, scale=(0.85, 1.0), interpolation=interpolation)
        ops = [crop, transforms.RandomHorizontalFlip()]
    else:
        ops = [transforms.Resize(224, interpolation=interpolation), transforms.CenterCrop(224)]
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    return transforms.Compose(ops)


def prepare_repo_import(
    repo_dir: str,
    extra_roots: Optional[Sequence[str]] = None,
    clear_module_prefixes: Optional[Sequence[str]] = None,
) -> None:
    repo_dir = os.path.abspath(repo_dir)
    roots: List[str] = [repo_dir]
    if extra_roots:
        roots.extend(os.path.abspath(path) for path in extra_roots if path)

    unique_roots: List[str] = []
    seen = set()
    for root in roots:
        if root and os.path.isdir(root) and root not in seen:
            seen.add(root)
            unique_roots.append(root)

    if clear_module_prefixes:
        prefixes = tuple(str(prefix) for prefix in clear_module_prefixes if prefix)
        if prefixes:
            dotted_prefixes = tuple(prefix + "." for prefix in prefixes)
            for module_name in list(sys.modules.keys()):
                if module_name in prefixes or module_name.startswith(dotted_prefixes):
                    sys.modules.pop(module_name, None)

    for root in reversed(unique_roots):
        if root not in sys.path:
            sys.path.insert(0, root)

    existing_pythonpath = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    merged_pythonpath: List[str] = []
    for root in unique_roots + existing_pythonpath:
        if root not in merged_pythonpath:
            merged_pythonpath.append(root)
    os.environ["PYTHONPATH"] = os.pathsep.join(merged_pythonpath)
    importlib.invalidate_caches()


def patchify_image_tensor(image_tensor: torch.Tensor, num_patch: int, test_mode: bool) -> torch.Tensor:
    image_tensor = image_tensor.unsqueeze(0)
    n_channels, kernel_h, kernel_w = 3, 224, 224
    step = 48 if (image_tensor.size(2) >= 1024 or image_tensor.size(3) >= 1024) else 32
    patches = (
        image_tensor.unfold(2, kernel_h, step)
        .unfold(3, kernel_w, step)
        .permute(2, 3, 0, 1, 4, 5)
        .reshape(-1, n_channels, kernel_h, kernel_w)
    )
    if patches.size(0) < num_patch:
        raise ValueError(f"Image is too small to extract {num_patch} patches.")
    if test_mode:
        sel_step = max(patches.size(0) // num_patch, 1)
        sel = torch.tensor([min(sel_step * i, patches.size(0) - 1) for i in range(num_patch)], dtype=torch.long)
    else:
        sel = torch.randint(low=0, high=patches.size(0), size=(num_patch,), dtype=torch.long)
    patches = patches[sel]
    resized = F.interpolate(image_tensor, size=(kernel_h, kernel_w), mode="bilinear", align_corners=False)
    return torch.cat([patches, resized], dim=0)


class IPCEPatchDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_base_dir: str,
        label_column: str,
        preprocess: transforms.Compose,
        num_patch: int,
        test_mode: bool,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.image_base_dir = image_base_dir
        self.label_column = label_column
        self.preprocess = preprocess
        self.num_patch = int(num_patch)
        self.test_mode = bool(test_mode)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]
        image = Image.open(os.path.join(self.image_base_dir, str(row["name"]))).convert("RGB")
        tensor = self.preprocess(image)
        patches = patchify_image_tensor(tensor, self.num_patch, self.test_mode)
        return {
            "I": patches,
            "mos": float(row[self.label_column]),
            "prompt": str(row["prompt"]),
            "image_name": str(row["name"]),
        }


def clip_agiqa_label_bin(mos_value: float) -> int:
    if 0 <= mos_value < 1.231:
        return 0
    if 1.231 <= mos_value < 2.266:
        return 1
    if 2.266 <= mos_value < 2.749:
        return 2
    if 2.749 <= mos_value < 3.092:
        return 3
    if 3.092 <= mos_value < 3.456:
        return 4
    return 5


class CLIPAGIQAImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_base_dir: str, transform: transforms.Compose, label_column: str) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.image_base_dir = image_base_dir
        self.transform = transform
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]
        image = Image.open(os.path.join(self.image_base_dir, str(row["name"]))).convert("RGB")
        return {
            "image": self.transform(image),
            "mos": float(row[self.label_column]),
            "label": clip_agiqa_label_bin(float(row[self.label_column])),
            "name": str(row["name"]),
        }


def do_ipce_batch_prompt(model: nn.Module, images: torch.Tensor, tokenized_texts: torch.Tensor) -> torch.Tensor:
    batch_size = images.size(0)
    num_patch = images.size(1)
    flat_images = images.view(-1, images.size(2), images.size(3), images.size(4))
    logits_per_image, _ = model.forward(flat_images, tokenized_texts)
    logits_per_image = logits_per_image.view(batch_size, num_patch, -1)
    logits_per_image = torch.cat(
        [logits_per_image[:, : num_patch - 1, :].mean(1, keepdim=True), logits_per_image[:, -1, :].unsqueeze(1)],
        dim=1,
    )
    prompt_scores = logits_per_image[0, :, :5].unsqueeze(0)
    if batch_size > 1:
        for idx in range(1, batch_size):
            prompt_scores = torch.cat(
                [prompt_scores, logits_per_image[idx, :, idx * 5 : (idx + 1) * 5].unsqueeze(0)],
                dim=0,
            )
    return F.softmax(prompt_scores, dim=2).mean(1)


def weighted_mos_from_logits(prob: torch.Tensor) -> torch.Tensor:
    score = sum((idx + 1) * prob[:, idx] for idx in range(5))
    return ((score - 1.0) / 4.0) * 5.0


def run_ipce_epoch(
    model: nn.Module,
    clip_module: Any,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> Tuple[float, List[str], List[float], List[float]]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    step_count = 0
    names: List[str] = []
    preds: List[float] = []
    targets: List[float] = []
    for batch in loader:
        images = batch["I"].to(device)
        mos = batch["mos"].to(device=device, dtype=torch.float32)
        prompt = list(batch["prompt"])
        texts = [f"a photo that {word} matches '{p}'" for p, word in product(prompt, QUALITY_PROMPT_WORDS)]
        input_texts = torch.cat([clip_module.tokenize(text, truncate=True) for text in texts]).to(device)
        if train_mode:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train_mode):
            prob = do_ipce_batch_prompt(model, images, input_texts)
            pred = weighted_mos_from_logits(prob)
            loss = torch.mean(torch.abs(pred - mos))
            if train_mode:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach().item())
        step_count += 1
        names.extend(list(batch["image_name"]))
        preds.extend(pred.detach().cpu().tolist())
        targets.extend(mos.detach().cpu().tolist())
    avg_loss = total_loss / max(step_count, 1)
    return avg_loss, names, preds, targets


def build_clip_agiqa_cfg(backbone_name: str = "ViT-B/16") -> Any:
    return types.SimpleNamespace(
        MODEL=types.SimpleNamespace(BACKBONE=types.SimpleNamespace(NAME=backbone_name)),
        INPUT=types.SimpleNamespace(SIZE=(224, 224)),
        TRAINER=types.SimpleNamespace(
            CLIP_AGIQA=types.SimpleNamespace(
                N_CTX=16,
                CTX_INIT="",
                CSC=False,
                CLASS_TOKEN_POSITION="end",
            )
        ),
    )


def init_clip_agiqa_state(repo_dir: str, device: torch.device, lr: float) -> Dict[str, Any]:
    repo_dir = os.path.abspath(repo_dir)
    repo_parent = os.path.dirname(repo_dir)
    extra_roots = [
        os.path.join(repo_parent, "Dassl.pytorch"),
        os.path.join(repo_parent, "CoOp"),
    ]
    prepare_repo_import(
        repo_dir,
        extra_roots=extra_roots,
        clear_module_prefixes=["trainers", "clip", "dassl", "datasets"],
    )
    from trainers.clip_agiqa import CustomCLIP, load_clip_to_cpu  # type: ignore

    cfg = build_clip_agiqa_cfg()
    classnames = ["terrible", "bad", "poor", "average", "good", "perfect"]
    clip_model = load_clip_to_cpu(cfg).float()
    model = CustomCLIP(cfg, classnames, clip_model)
    for name, param in model.named_parameters():
        if ("regression" not in name) and ("prompt_learner" not in name):
            param.requires_grad_(False)
    model.to(device)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    return {"model": model, "optimizer": optimizer, "scheduler": scheduler}


def run_clip_agiqa_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    device: torch.device,
) -> Tuple[float, List[str], List[float], List[float]]:
    train_mode = optimizer is not None
    model.train(train_mode)
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    step_count = 0
    names: List[str] = []
    preds: List[float] = []
    targets: List[float] = []
    for batch in loader:
        images = batch["image"].to(device)
        mos = batch["mos"].to(device=device, dtype=torch.float32)
        if train_mode:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train_mode):
            pred = model(images).float()
            loss = loss_fn(pred, mos)
            if train_mode:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach().item())
        step_count += 1
        names.extend(list(batch["name"]))
        preds.extend(pred.detach().cpu().tolist())
        targets.extend(mos.detach().cpu().tolist())
    if train_mode and scheduler is not None:
        scheduler.step()
    avg_loss = total_loss / max(step_count, 1)
    return avg_loss, names, preds, targets


def concat_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    valid = [frame.copy() for frame in frames if not frame.empty]
    if not valid:
        return frames[0].iloc[0:0].copy() if frames else pd.DataFrame()
    return pd.concat(valid, ignore_index=True)


def save_checkpoint_stub(path: str) -> None:
    torch.save({"ok": True}, path)


def load_yaml_config(path: str) -> Dict[str, Any]:
    import yaml  # type: ignore

    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML config at {path} must be a mapping.")
    return loaded


def dump_yaml_config(path: str, payload: Dict[str, Any]) -> None:
    import yaml  # type: ignore

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def ensure_symlink_or_copy(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        return
    dst_parent = os.path.dirname(dst)
    if dst_parent:
        os.makedirs(dst_parent, exist_ok=True)
    try:
        os.symlink(src, dst, target_is_directory=os.path.isdir(src))
    except Exception:
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def extract_record_name(record: Dict[str, Any]) -> str:
    image_value = record.get("image")
    if isinstance(image_value, str) and image_value.strip():
        return os.path.basename(image_value)
    images_value = record.get("images")
    if isinstance(images_value, list) and images_value:
        return os.path.basename(str(images_value[0]))
    raise ValueError("Unable to resolve image name from M3 record.")


def filter_records_by_names(records: Sequence[Dict[str, Any]], names: Sequence[str]) -> List[Dict[str, Any]]:
    wanted = {str(name) for name in names}
    filtered: List[Dict[str, Any]] = []
    for record in records:
        try:
            record_name = extract_record_name(record)
        except ValueError:
            continue
        if record_name in wanted:
            filtered.append(copy.deepcopy(record))
    return filtered


def rewrite_m3_record_paths(records: Sequence[Dict[str, Any]], image_dir: str) -> List[Dict[str, Any]]:
    rewritten: List[Dict[str, Any]] = []
    for record in records:
        item = copy.deepcopy(record)
        record_name = extract_record_name(item)
        canonical = os.path.join(image_dir, record_name)
        if "image" in item:
            item["image"] = canonical
        if "images" in item:
            item["images"] = [canonical]
        rewritten.append(item)
    return rewritten


def parse_epoch_from_checkpoint_name(path: str, fallback: int) -> int:
    base = os.path.basename(path)
    marker = "epoch="
    if marker in base:
        value = base.split(marker, 1)[1].split("-", 1)[0]
        try:
            return int(value) + 1
        except ValueError:
            pass
    return int(fallback)


def build_ma_transforms(repo_dir: str) -> Tuple[Any, Any]:
    prepare_repo_import(repo_dir)
    from utils.process import Normalize, RandCrop, RandHorizontalFlip, RandRotation, ToTensor  # type: ignore

    train_transform = transforms.Compose(
        [
            RandRotation(prob_aug=0.7),
            RandCrop(patch_size=224),
            Normalize(0.5, 0.5),
            RandHorizontalFlip(prob_aug=0.7),
            ToTensor(),
        ]
    )
    eval_transform = transforms.Compose(
        [
            RandCrop(patch_size=224),
            Normalize(0.5, 0.5),
            ToTensor(),
        ]
    )
    return train_transform, eval_transform


def build_ma_dataset(df: pd.DataFrame, image_base_dir: str, tensor_root: str, transform: Any, repo_dir: str) -> Dataset:
    prepare_repo_import(repo_dir)
    from data.AIGC_general import AIGCgeneral  # type: ignore

    labels = df["__adapter_label__"].astype(float).tolist()
    names = df["name"].astype(str).tolist()
    return AIGCgeneral(
        dis_path=image_base_dir,
        labels=labels,
        pic_names=names,
        transform=transform,
        keep_ratio=1.0,
        tensor_root=tensor_root,
    )


def run_ma_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    device: torch.device,
) -> Tuple[float, List[str], List[float], List[float]]:
    train_mode = optimizer is not None
    model.train(train_mode)
    criterion = nn.MSELoss()
    losses: List[float] = []
    names: List[str] = []
    preds: List[float] = []
    targets: List[float] = []
    for batch in loader:
        x_d = batch["d_img_org"].to(device)
        labels = torch.squeeze(batch["score"].type(torch.FloatTensor)).to(device)
        tensor1 = batch["tensor_1"].to(device)
        tensor2 = batch["tensor_2"].to(device)
        if train_mode:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train_mode):
            pred = model(x_d, tensor1=tensor1, tensor2=tensor2)
            loss = criterion(torch.squeeze(pred), labels)
            if train_mode:
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
        losses.append(float(loss.detach().item()))
        names.extend(list(batch["name_"]))
        preds.extend(torch.squeeze(pred).detach().cpu().reshape(-1).tolist())
        targets.extend(labels.detach().cpu().reshape(-1).tolist())
    return float(np.mean(losses) if losses else 0.0), names, preds, targets


def build_m3_workspace(args: argparse.Namespace) -> Dict[str, str]:
    workspace_root = os.path.join(args.output_dir, "m3_workspace")
    dataset_root = os.path.join(workspace_root, "data", args.dataset_name)
    images_link = os.path.join(dataset_root, "images")
    os.makedirs(dataset_root, exist_ok=True)
    ensure_symlink_or_copy(args.image_base_dir, images_link)
    return {
        "workspace_root": workspace_root,
        "data_root": os.path.join(workspace_root, "data"),
        "dataset_root": dataset_root,
        "images_link": images_link,
    }


def write_m3_descriptor(dataset_root: str, frames: Sequence[pd.DataFrame]) -> str:
    descriptor = concat_frames(frames)
    descriptor.to_csv(os.path.join(dataset_root, "data.csv"), index=False, encoding="utf-8")
    return os.path.join(dataset_root, "data.csv")


def write_m3_pool_subset(src_json: Any, names: Sequence[str], image_dir: str, out_json: str) -> None:
    source_paths = [src_json] if isinstance(src_json, str) else list(src_json)
    all_records: List[Dict[str, Any]] = []
    for path in source_paths:
        with open(path, "r", encoding="utf-8") as f:
            all_records.extend(json.load(f))
    filtered = filter_records_by_names(all_records, names)
    expected = {str(name) for name in names}
    found = {extract_record_name(record) for record in filtered}
    missing = sorted(expected - found)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"M3 pool json 缺少 {len(missing)} 个样本，例如: {preview}")
    rewritten = rewrite_m3_record_paths(filtered, image_dir=image_dir)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rewritten, f, ensure_ascii=False, indent=2)


def find_m3_checkpoint(repo_dir: str, model_name: str, run_name: str, prefer_last: bool) -> str:
    ckpt_dir = os.path.join(repo_dir, "checkpoints", model_name, run_name)
    if prefer_last:
        last_path = os.path.join(ckpt_dir, "last.ckpt")
        if os.path.exists(last_path):
            return last_path
    candidates = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {ckpt_dir}")
    if prefer_last:
        return candidates[-1]
    best_candidates = [path for path in candidates if os.path.basename(path).startswith("best-")]
    return best_candidates[-1] if best_candidates else candidates[-1]


def run_m3_native(repo_dir: str, config_path: str) -> None:
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    subprocess.run([sys.executable, "main.py", "--config", config_path], cwd=repo_dir, check=True, env=env)


def read_m3_prediction_json(repo_dir: str, model_name: str, run_name: str) -> Dict[str, float]:
    prediction_path = os.path.join(repo_dir, "predictions", model_name, f"{run_name}.json")
    with open(prediction_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(name): float(value) for name, value in raw.items()}


def run_ipce_backend(args: argparse.Namespace) -> Dict[str, Any]:
    prepare_repo_import(args.repo_dir)
    import clip  # type: ignore

    device = torch.device(args.device)
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)
    max_epochs = args.epochs if args.epochs is not None else 30
    batch_size = args.batch_size if args.batch_size is not None else 32
    lr = args.lr if args.lr is not None else 5e-6

    train_loader = DataLoader(
        IPCEPatchDataset(train_df, args.image_base_dir, args.label_column, build_ipce_transform(train=True), 8, False),
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        IPCEPatchDataset(val_df, args.image_base_dir, args.label_column, build_ipce_transform(train=False), 15, True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        IPCEPatchDataset(test_df, args.image_base_dir, args.label_column, build_ipce_transform(train=False), 15, True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model, _ = clip.load("ViT-B/32", device=device, jit=False)
    model = model.float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)

    history: List[Dict[str, Any]] = []
    best_epoch = max_epochs
    best_score = float("-inf")
    best_state = snapshot_state_dict(model)
    best_val_names: List[str] = []
    best_val_preds: List[float] = []
    selection_based_on = "validation"
    selection_metric = "srocc"
    selection_score = float("-inf")
    stopped_on_train_loss_threshold = False

    if len(val_df) == 0:
        for epoch in range(1, max_epochs + 1):
            train_loss, _, _, _ = run_ipce_epoch(model, clip, train_loader, optimizer, device)
            history.append({"epoch": epoch, "train_loss": train_loss})
            best_epoch = epoch
            selection_score = -float(train_loss)
            if (
                args.train_loss_stop_threshold is not None
                and float(train_loss) <= float(args.train_loss_stop_threshold)
            ):
                stopped_on_train_loss_threshold = True
                selection_based_on = "train_loss_threshold"
                selection_metric = "train_loss"
                break
        if not stopped_on_train_loss_threshold:
            selection_based_on = "no_validation_last_epoch"
            selection_metric = "train_loss"
        best_state = snapshot_state_dict(model)
    else:
        for epoch in range(1, max_epochs + 1):
            train_loss, _, _, _ = run_ipce_epoch(model, clip, train_loader, optimizer, device)
            val_loss, val_names, val_preds, val_targets = run_ipce_epoch(model, clip, val_loader, None, device)
            val_srocc = safe_srocc(val_targets, val_preds)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_srocc": val_srocc})
            if val_srocc >= best_score:
                best_score = val_srocc
                selection_score = float(val_srocc)
                best_epoch = epoch
                best_state = snapshot_state_dict(model)
                best_val_names = list(val_names)
                best_val_preds = list(val_preds)

    if args.refit_on_trainval and len(val_df) > 0:
        refit_df = concat_frames([train_df, val_df])
        refit_loader = DataLoader(
            IPCEPatchDataset(refit_df, args.image_base_dir, args.label_column, build_ipce_transform(train=True), 8, False),
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        model, _ = clip.load("ViT-B/32", device=device, jit=False)
        model = model.float().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
        for _ in range(max(best_epoch, 1)):
            run_ipce_epoch(model, clip, refit_loader, optimizer, device)
    else:
        load_snapshot(model, best_state)

    torch.save({"model_state_dict": model.state_dict(), "best_epoch": best_epoch}, os.path.join(args.output_dir, "checkpoint.pt"))
    if len(val_df) == 0:
        save_prediction_csv(os.path.join(args.output_dir, "val_preds_raw.csv"), [], [])
    else:
        temp_model, _ = clip.load("ViT-B/32", device=device, jit=False)
        temp_model = temp_model.float().to(device)
        load_snapshot(temp_model, best_state)
        _, val_names, val_preds, _ = run_ipce_epoch(temp_model, clip, val_loader, None, device)
        best_val_names = val_names
        best_val_preds = val_preds
        save_prediction_csv(os.path.join(args.output_dir, "val_preds_raw.csv"), best_val_names, best_val_preds)

    _, test_names, test_preds, _ = run_ipce_epoch(model, clip, test_loader, None, device)
    save_prediction_csv(os.path.join(args.output_dir, "test_preds_raw.csv"), test_names, test_preds)
    return {
        "best_epoch": int(max(best_epoch, 1)),
        "history": history,
        "trainable_params": count_parameters(model, trainable_only=True),
        "total_params": count_parameters(model, trainable_only=False),
        "refit_on_trainval": bool(args.refit_on_trainval and len(val_df) > 0),
        "epochs": int(max_epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "selection_based_on": str(selection_based_on),
        "selection_metric": str(selection_metric),
        "selection_score": float(selection_score if np.isfinite(selection_score) else 0.0),
        "train_loss_stop_threshold": (
            None if args.train_loss_stop_threshold is None else float(args.train_loss_stop_threshold)
        ),
        "stopped_on_train_loss_threshold": bool(stopped_on_train_loss_threshold),
    }


def run_clip_agiqa_backend(args: argparse.Namespace) -> Dict[str, Any]:
    if args.task != "quality":
        raise ValueError("CLIP-AGIQA 当前只支持 quality 公平重跑。")
    device = torch.device(args.device)
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)
    max_epochs = args.epochs if args.epochs is not None else 100
    batch_size = args.batch_size if args.batch_size is not None else 32
    lr = args.lr if args.lr is not None else 0.002

    train_loader = DataLoader(
        CLIPAGIQAImageDataset(train_df, args.image_base_dir, build_clip_agiqa_transform(train=True), args.label_column),
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        CLIPAGIQAImageDataset(val_df, args.image_base_dir, build_clip_agiqa_transform(train=False), args.label_column),
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        CLIPAGIQAImageDataset(test_df, args.image_base_dir, build_clip_agiqa_transform(train=False), args.label_column),
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    state = init_clip_agiqa_state(args.repo_dir, device, lr)
    model = state["model"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]

    history: List[Dict[str, Any]] = []
    best_epoch = max_epochs
    best_score = float("-inf")
    best_state = snapshot_state_dict(model)
    selection_based_on = "validation"
    selection_metric = "srocc"
    selection_score = float("-inf")
    stopped_on_train_loss_threshold = False

    if len(val_df) == 0:
        for epoch in range(1, max_epochs + 1):
            train_loss, _, _, _ = run_clip_agiqa_epoch(model, train_loader, optimizer, scheduler, device)
            history.append({"epoch": epoch, "train_loss": train_loss})
            best_epoch = epoch
            selection_score = -float(train_loss)
            if (
                args.train_loss_stop_threshold is not None
                and float(train_loss) <= float(args.train_loss_stop_threshold)
            ):
                stopped_on_train_loss_threshold = True
                selection_based_on = "train_loss_threshold"
                selection_metric = "train_loss"
                break
        if not stopped_on_train_loss_threshold:
            selection_based_on = "no_validation_last_epoch"
            selection_metric = "train_loss"
        best_state = snapshot_state_dict(model)
        save_prediction_csv(os.path.join(args.output_dir, "val_preds_raw.csv"), [], [])
    else:
        for epoch in range(1, max_epochs + 1):
            train_loss, _, _, _ = run_clip_agiqa_epoch(model, train_loader, optimizer, scheduler, device)
            val_loss, val_names, val_preds, val_targets = run_clip_agiqa_epoch(model, val_loader, None, None, device)
            val_srocc = safe_srocc(val_targets, val_preds)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_srocc": val_srocc})
            if val_srocc >= best_score:
                best_score = val_srocc
                selection_score = float(val_srocc)
                best_epoch = epoch
                best_state = snapshot_state_dict(model)
                save_prediction_csv(os.path.join(args.output_dir, "val_preds_raw.csv"), val_names, val_preds)

    if args.refit_on_trainval and len(val_df) > 0:
        refit_df = concat_frames([train_df, val_df])
        refit_loader = DataLoader(
            CLIPAGIQAImageDataset(refit_df, args.image_base_dir, build_clip_agiqa_transform(train=True), args.label_column),
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        state = init_clip_agiqa_state(args.repo_dir, device, lr)
        model = state["model"]
        optimizer = state["optimizer"]
        scheduler = state["scheduler"]
        for _ in range(max(best_epoch, 1)):
            run_clip_agiqa_epoch(model, refit_loader, optimizer, scheduler, device)
    else:
        load_snapshot(model, best_state)

    torch.save({"model_state_dict": model.state_dict(), "best_epoch": best_epoch}, os.path.join(args.output_dir, "checkpoint.pt"))
    _, test_names, test_preds, _ = run_clip_agiqa_epoch(model, test_loader, None, None, device)
    save_prediction_csv(os.path.join(args.output_dir, "test_preds_raw.csv"), test_names, test_preds)
    return {
        "best_epoch": int(max(best_epoch, 1)),
        "history": history,
        "trainable_params": count_parameters(model, trainable_only=True),
        "total_params": count_parameters(model, trainable_only=False),
        "refit_on_trainval": bool(args.refit_on_trainval and len(val_df) > 0),
        "epochs": int(max_epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "selection_based_on": str(selection_based_on),
        "selection_metric": str(selection_metric),
        "selection_score": float(selection_score if np.isfinite(selection_score) else 0.0),
        "train_loss_stop_threshold": (
            None if args.train_loss_stop_threshold is None else float(args.train_loss_stop_threshold)
        ),
        "stopped_on_train_loss_threshold": bool(stopped_on_train_loss_threshold),
    }


def run_ma_agiqa_backend(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.tensor_root:
        raise ValueError("MA-AGIQA 需要 --tensor_root。")
    prepare_repo_import(args.repo_dir)
    from models.MA_AGIQA import MA_AGIQA  # type: ignore

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("MA-AGIQA 当前仅支持 CUDA 设备。")
    train_df = pd.read_csv(args.train_csv).copy()
    val_df = pd.read_csv(args.val_csv).copy()
    test_df = pd.read_csv(args.test_csv).copy()
    train_df["__adapter_label__"] = pd.to_numeric(train_df[args.label_column], errors="coerce").fillna(0.0)
    val_df["__adapter_label__"] = pd.to_numeric(val_df[args.label_column], errors="coerce").fillna(0.0)
    test_df["__adapter_label__"] = pd.to_numeric(test_df[args.label_column], errors="coerce").fillna(0.0)

    max_epochs = args.epochs if args.epochs is not None else 30
    batch_size = args.batch_size if args.batch_size is not None else 8
    lr = args.lr if args.lr is not None else 1e-5
    train_transform, eval_transform = build_ma_transforms(args.repo_dir)

    def _build_model() -> nn.Module:
        model = MA_AGIQA(
            embed_dim=768,
            num_outputs=1,
            dim_mlp=768,
            patch_size=8,
            img_size=224,
            window_size=4,
            depths=[2, 2],
            num_heads=[4, 4],
            num_tab=2,
            scale=0.8,
        )
        return model.to(device)

    def _build_optimizer(model: nn.Module) -> Tuple[Any, Any]:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0)
        return optimizer, scheduler

    train_loader = DataLoader(
        build_ma_dataset(train_df, args.image_base_dir, args.tensor_root, train_transform, args.repo_dir),
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        build_ma_dataset(val_df, args.image_base_dir, args.tensor_root, eval_transform, args.repo_dir),
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        build_ma_dataset(test_df, args.image_base_dir, args.tensor_root, eval_transform, args.repo_dir),
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    model = _build_model()
    optimizer, scheduler = _build_optimizer(model)
    history: List[Dict[str, Any]] = []
    best_epoch = max_epochs
    best_score = float("-inf")
    best_state = snapshot_state_dict(model)
    selection_based_on = "validation"
    selection_metric = "srocc"
    selection_score = float("-inf")
    stopped_on_train_loss_threshold = False

    if len(val_df) == 0:
        for epoch in range(1, max_epochs + 1):
            train_loss, _, _, _ = run_ma_epoch(model, train_loader, optimizer, scheduler, device)
            history.append({"epoch": epoch, "train_loss": train_loss})
            best_epoch = epoch
            selection_score = -float(train_loss)
            if (
                args.train_loss_stop_threshold is not None
                and float(train_loss) <= float(args.train_loss_stop_threshold)
            ):
                stopped_on_train_loss_threshold = True
                selection_based_on = "train_loss_threshold"
                selection_metric = "train_loss"
                break
        if not stopped_on_train_loss_threshold:
            selection_based_on = "no_validation_last_epoch"
            selection_metric = "train_loss"
        best_state = snapshot_state_dict(model)
        save_prediction_csv(os.path.join(args.output_dir, "val_preds_raw.csv"), [], [])
    else:
        for epoch in range(1, max_epochs + 1):
            train_loss, _, _, _ = run_ma_epoch(model, train_loader, optimizer, scheduler, device)
            val_loss, val_names, val_preds, val_targets = run_ma_epoch(model, val_loader, None, None, device)
            val_srocc = safe_srocc(val_targets, val_preds)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_srocc": val_srocc})
            if val_srocc >= best_score:
                best_score = val_srocc
                selection_score = float(val_srocc)
                best_epoch = epoch
                best_state = snapshot_state_dict(model)
                save_prediction_csv(os.path.join(args.output_dir, "val_preds_raw.csv"), val_names, val_preds)

    if args.refit_on_trainval and len(val_df) > 0:
        refit_df = concat_frames([train_df, val_df])
        refit_loader = DataLoader(
            build_ma_dataset(refit_df, args.image_base_dir, args.tensor_root, train_transform, args.repo_dir),
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )
        model = _build_model()
        optimizer, scheduler = _build_optimizer(model)
        for _ in range(max(best_epoch, 1)):
            run_ma_epoch(model, refit_loader, optimizer, scheduler, device)
    else:
        load_snapshot(model, best_state)

    torch.save({"model_state_dict": model.state_dict(), "best_epoch": best_epoch}, os.path.join(args.output_dir, "checkpoint.pt"))
    _, test_names, test_preds, _ = run_ma_epoch(model, test_loader, None, None, device)
    save_prediction_csv(os.path.join(args.output_dir, "test_preds_raw.csv"), test_names, test_preds)
    return {
        "best_epoch": int(max(best_epoch, 1)),
        "history": history,
        "trainable_params": count_parameters(model, trainable_only=True),
        "total_params": count_parameters(model, trainable_only=False),
        "refit_on_trainval": bool(args.refit_on_trainval and len(val_df) > 0),
        "epochs": int(max_epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "tensor_root": args.tensor_root,
        "selection_based_on": str(selection_based_on),
        "selection_metric": str(selection_metric),
        "selection_score": float(selection_score if np.isfinite(selection_score) else 0.0),
        "train_loss_stop_threshold": (
            None if args.train_loss_stop_threshold is None else float(args.train_loss_stop_threshold)
        ),
        "stopped_on_train_loss_threshold": bool(stopped_on_train_loss_threshold),
    }


def run_m3_agiqa_backend(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.base_config:
        raise ValueError("M3-AGIQA 需要 --base_config。")
    for required_path in [args.train_pool_json, args.eval_pool_json, args.test_pool_json]:
        if not required_path:
            raise ValueError("M3-AGIQA 需要 train/eval/test 三个 processed json 池路径。")
    if torch.device(args.device).type != "cuda":
        raise ValueError("M3-AGIQA 当前仅支持 CUDA 设备。")

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)
    workspace = build_m3_workspace(args)
    write_m3_descriptor(workspace["dataset_root"], [train_df, val_df, test_df])
    all_pool_sources = [args.train_pool_json, args.eval_pool_json, args.test_pool_json]

    train_json = os.path.join(workspace["workspace_root"], "train_pool_filtered.json")
    val_json = os.path.join(workspace["workspace_root"], "val_pool_filtered.json")
    test_json = os.path.join(workspace["workspace_root"], "test_pool_filtered.json")
    # The released M3 processed json files are tied to the authors' original
    # split. For our fair re-split protocol, any requested sample may reside in
    # any of those three pools, so we resolve each subset from their union.
    write_m3_pool_subset(all_pool_sources, train_df["name"].tolist(), workspace["images_link"], train_json)
    write_m3_pool_subset(all_pool_sources, val_df["name"].tolist(), workspace["images_link"], val_json)
    write_m3_pool_subset(all_pool_sources, test_df["name"].tolist(), workspace["images_link"], test_json)

    base_cfg = load_yaml_config(args.base_config)
    model_name = str(base_cfg.get("model", "minicpm-xlstm"))
    max_epochs = args.epochs if args.epochs is not None else int(base_cfg.get("max_epochs", 20))
    batch_size = args.batch_size if args.batch_size is not None else int(base_cfg.get("batch_size", 1))
    lr = args.lr if args.lr is not None else float(base_cfg.get("lr", 1e-5))
    run_name = f"fair_{args.task}_{args.seed}_{Path(args.output_dir).name}"

    train_cfg = dict(base_cfg)
    train_cfg.update(
        {
            "dataset": args.dataset_name,
            "label_name": args.label_column,
            "data_dir": workspace["data_root"],
            "data_path": train_json,
            "eval_data_res_path": val_json,
            "test_data_path": test_json,
            "run_name": run_name,
            "seed": int(args.seed),
            "max_epochs": int(max_epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "wandb": False,
            "stage": "train",
        }
    )
    train_cfg_path = os.path.join(workspace["workspace_root"], "m3_train.yaml")
    dump_yaml_config(train_cfg_path, train_cfg)
    run_m3_native(args.repo_dir, train_cfg_path)
    best_ckpt = find_m3_checkpoint(args.repo_dir, model_name, run_name, prefer_last=False)
    best_epoch = parse_epoch_from_checkpoint_name(best_ckpt, fallback=max_epochs)

    ckpt_for_predict = best_ckpt
    refit_performed = False
    if args.refit_on_trainval and len(val_df) > 0:
        merged_train_json = os.path.join(workspace["workspace_root"], "trainval_pool_filtered.json")
        merged_names = concat_frames([train_df, val_df])["name"].tolist()
        write_m3_pool_subset(all_pool_sources, merged_names, workspace["images_link"], merged_train_json)
        refit_run_name = run_name + "_refit"
        refit_cfg = dict(train_cfg)
        refit_cfg.update(
            {
                "data_path": merged_train_json,
                "run_name": refit_run_name,
                "max_epochs": int(max(best_epoch, 1)),
                "stage": "train",
            }
        )
        refit_cfg_path = os.path.join(workspace["workspace_root"], "m3_refit.yaml")
        dump_yaml_config(refit_cfg_path, refit_cfg)
        run_m3_native(args.repo_dir, refit_cfg_path)
        ckpt_for_predict = find_m3_checkpoint(args.repo_dir, model_name, refit_run_name, prefer_last=True)
        refit_performed = True

    def _run_predict(split_json: str, suffix: str) -> Dict[str, float]:
        predict_cfg = dict(train_cfg)
        predict_cfg.update(
            {
                "run_name": f"{run_name}_{suffix}",
                "ckpt_path": ckpt_for_predict,
                "test_data_path": split_json,
                "stage": "predict",
            }
        )
        predict_cfg_path = os.path.join(workspace["workspace_root"], f"m3_predict_{suffix}.yaml")
        dump_yaml_config(predict_cfg_path, predict_cfg)
        run_m3_native(args.repo_dir, predict_cfg_path)
        return read_m3_prediction_json(args.repo_dir, model_name, predict_cfg["run_name"])

    if len(val_df) == 0:
        save_prediction_csv(os.path.join(args.output_dir, "val_preds_raw.csv"), [], [])
    else:
        val_pred_map = _run_predict(val_json, "val")
        val_names = [str(name) for name in val_df["name"].tolist()]
        save_prediction_csv(
            os.path.join(args.output_dir, "val_preds_raw.csv"),
            val_names,
            [float(val_pred_map[name]) for name in val_names],
        )

    test_pred_map = _run_predict(test_json, "test")
    test_names = [str(name) for name in test_df["name"].tolist()]
    save_prediction_csv(
        os.path.join(args.output_dir, "test_preds_raw.csv"),
        test_names,
        [float(test_pred_map[name]) for name in test_names],
    )
    save_checkpoint_stub(os.path.join(args.output_dir, "checkpoint.pt"))
    return {
        "best_epoch": int(max(best_epoch, 1)),
        "history": [],
        "trainable_params": 0,
        "total_params": 0,
        "refit_on_trainval": bool(refit_performed),
        "epochs": int(max_epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "best_checkpoint_path": ckpt_for_predict,
        "base_config": args.base_config,
    }


def main() -> None:
    parser = argparse.ArgumentParser("External baseline adapter")
    parser.add_argument("--backend", required=True, choices=["ipce", "clip_agiqa", "ma_agiqa", "m3_agiqa"])
    parser.add_argument("--repo_dir", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--image_base_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task", required=True, choices=["quality", "alignment"])
    parser.add_argument("--label_column", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--refit_on_trainval", action="store_true")
    parser.add_argument("--train_loss_stop_threshold", type=float, default=None)
    parser.add_argument("--tensor_root", default="")
    parser.add_argument("--base_config", default="")
    parser.add_argument("--train_pool_json", default="")
    parser.add_argument("--eval_pool_json", default="")
    parser.add_argument("--test_pool_json", default="")
    parser.add_argument("--dataset_name", default="agiqa-3k")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_global_seed(args.seed)
    start_time = time.time()

    if args.backend == "ipce":
        payload = run_ipce_backend(args)
    elif args.backend == "clip_agiqa":
        payload = run_clip_agiqa_backend(args)
    elif args.backend == "ma_agiqa":
        payload = run_ma_agiqa_backend(args)
    elif args.backend == "m3_agiqa":
        payload = run_m3_agiqa_backend(args)
    else:
        raise ValueError(f"Unsupported backend='{args.backend}'.")

    payload.update(
        {
            "backend": args.backend,
            "task": args.task,
            "seed": int(args.seed),
            "runtime_sec": float(time.time() - start_time),
        }
    )
    with open(os.path.join(args.output_dir, "adapter_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
