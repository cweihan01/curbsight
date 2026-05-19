"""
Ground-truth loading and occupancy validation metrics for parking_management.

GT CSV columns: spot_id, start_frame, end_frame, status (occupied|available|unknown).
Spot index i in bounding_boxes.json maps to spot_id str(i + 1).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, TextIO


def normalize_gt_status(raw: str) -> str:
    s = raw.strip().lower()
    aliases = {"free": "available", "empty": "available", "vacant": "available"}
    s = aliases.get(s, s)
    if s not in ("occupied", "available", "unknown"):
        raise ValueError(f"Invalid GT status {raw!r}; expected occupied, available, or unknown.")
    return s


_GT_REQUIRED_CANONICAL = frozenset({"spot_id", "start_frame", "end_frame", "status"})


def _normalize_gt_header(name: str) -> str:
    return name.lstrip("\ufeff").strip().lower()


def _gt_column_keymap(fieldnames: Iterable[str] | None) -> dict[str, str]:
    keymap: dict[str, str] = {}
    if not fieldnames:
        return keymap
    for raw in fieldnames:
        canon = _normalize_gt_header(raw)
        if canon and canon not in keymap:
            keymap[canon] = raw
    return keymap


def load_gt_intervals(path: Path) -> dict[str, list[tuple[int, int, str]]]:
    """Load GT intervals keyed by spot_id (string). Validates non-overlapping intervals per spot."""
    by_spot: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or invalid CSV: {path}")
        keymap = _gt_column_keymap(reader.fieldnames)
        missing = _GT_REQUIRED_CANONICAL - set(keymap.keys())
        if missing:
            shown = [_normalize_gt_header(h) for h in reader.fieldnames]
            raise ValueError(
                f"GT CSV missing columns {missing}; "
                f"normalized_headers={shown!r} raw_fieldnames={reader.fieldnames!r}"
            )

        def get_cell(r: dict[str, Any], canonical: str) -> str:
            raw_key = keymap[canonical]
            v = r.get(raw_key)
            return "" if v is None else str(v)

        for row in reader:
            spot_id = get_cell(row, "spot_id").strip()
            if not spot_id:
                raise ValueError(f"Empty spot_id in GT row: {row}")
            start = int(get_cell(row, "start_frame"))
            end = int(get_cell(row, "end_frame"))
            status = normalize_gt_status(get_cell(row, "status"))
            if start > end:
                raise ValueError(f"Invalid interval start>end: {row}")
            by_spot[spot_id].append((start, end, status))

    for key, intervals in by_spot.items():
        intervals.sort(key=lambda t: (t[0], t[1]))
        for i in range(len(intervals) - 1):
            a0, a1, _ = intervals[i]
            b0, b1, _ = intervals[i + 1]
            if max(a0, b0) <= min(a1, b1):
                raise ValueError(
                    f"Overlapping GT intervals for {key}: {intervals[i]} vs {intervals[i + 1]}"
                )

    return dict(by_spot)


def gt_lookup(
    intervals: list[tuple[int, int, str]],
    frame_index: int,
) -> tuple[str | None, str | None]:
    """Return (gt_label, skip_reason). skip_reason is gt_unknown, no_gt, or None."""
    hit = [(s, e, st) for s, e, st in intervals if s <= frame_index <= e]
    if not hit:
        return None, "no_gt"
    if len(hit) > 1:
        raise ValueError(f"Multiple GT intervals cover frame {frame_index}: {hit}")
    _, _, st = hit[0]
    if st == "unknown":
        return None, "gt_unknown"
    return st, None


def spots_from_region_flags(flags: list[bool]) -> list[dict[str, str]]:
    """Map per-region booleans to spot_id/status dicts (1-based spot_id)."""
    return [
        {"spot_id": str(i + 1), "status": "occupied" if occ else "available"}
        for i, occ in enumerate(flags)
    ]


def _occupancy_metrics_from_counts(
    tp_occ: int, fp_occ: int, fn_occ: int, tn: int
) -> dict[str, Any]:
    n = tp_occ + tn + fp_occ + fn_occ
    acc = (tp_occ + tn) / n if n else None
    p_occ = tp_occ / (tp_occ + fp_occ) if (tp_occ + fp_occ) else None
    r_occ = tp_occ / (tp_occ + fn_occ) if (tp_occ + fn_occ) else None
    f1_occ = (
        2 * p_occ * r_occ / (p_occ + r_occ)
        if p_occ is not None and r_occ is not None and (p_occ + r_occ) > 0
        else None
    )
    return {
        "n_compared": n,
        "accuracy": acc,
        "precision_occupied": p_occ,
        "recall_occupied": r_occ,
        "f1_occupied": f1_occ,
        "confusion": {
            "tp_occupied": tp_occ,
            "tn_available": tn,
            "fp_pred_occupied_gt_available": fp_occ,
            "fn_pred_available_gt_occupied": fn_occ,
        },
    }


@dataclass
class ValidationStats:
    tp_occ: int = 0
    fp_occ: int = 0
    fn_occ: int = 0
    tn: int = 0
    n_skipped_gt_unknown: int = 0
    n_skipped_no_gt_interval: int = 0
    n_skipped_spot_not_in_gt: int = 0
    disagreements: list[dict[str, Any]] = field(default_factory=list)

    def record(self, *, gt: str, pred: str, meta: dict[str, Any]) -> None:
        if gt == "occupied" and pred == "occupied":
            self.tp_occ += 1
        elif gt == "available" and pred == "available":
            self.tn += 1
        elif gt == "occupied" and pred == "available":
            self.fn_occ += 1
            self.disagreements.append({**meta, "gt": gt, "pred": pred})
        elif gt == "available" and pred == "occupied":
            self.fp_occ += 1
            self.disagreements.append({**meta, "gt": gt, "pred": pred})
        else:
            raise ValueError(f"Unexpected gt/pred pair: {gt!r}, {pred!r}")

    def metrics_dict(self) -> dict[str, Any]:
        out = _occupancy_metrics_from_counts(self.tp_occ, self.fp_occ, self.fn_occ, self.tn)
        out["skips"] = {
            "gt_unknown": self.n_skipped_gt_unknown,
            "no_gt_interval": self.n_skipped_no_gt_interval,
            "spot_not_in_gt_csv": self.n_skipped_spot_not_in_gt,
        }
        return out


@dataclass
class PerSpotMetricsAccumulator:
    tp_occ: int = 0
    fp_occ: int = 0
    fn_occ: int = 0
    tn: int = 0
    n_skipped_gt_unknown: int = 0
    n_skipped_no_gt_interval: int = 0
    n_skipped_spot_not_in_gt: int = 0

    def record_match(self, *, gt: str, pred: str) -> None:
        if gt == "occupied" and pred == "occupied":
            self.tp_occ += 1
        elif gt == "available" and pred == "available":
            self.tn += 1
        elif gt == "occupied" and pred == "available":
            self.fn_occ += 1
        elif gt == "available" and pred == "occupied":
            self.fp_occ += 1
        else:
            raise ValueError(f"Unexpected gt/pred pair: {gt!r}, {pred!r}")

    def to_metrics_dict(self) -> dict[str, Any]:
        out = _occupancy_metrics_from_counts(self.tp_occ, self.fp_occ, self.fn_occ, self.tn)
        out["skips"] = {
            "gt_unknown": self.n_skipped_gt_unknown,
            "no_gt_interval": self.n_skipped_no_gt_interval,
            "spot_not_in_gt_csv": self.n_skipped_spot_not_in_gt,
        }
        return out


def _spot_sort_key(spot_id: str) -> tuple[int, str]:
    if spot_id.isdigit():
        return (int(spot_id), spot_id)
    return (10**9, spot_id)


def compare_spots_to_gt(
    spots: list[dict[str, str]],
    *,
    frame_index: int,
    inference_index: int,
    gt_by_spot: dict[str, list[tuple[int, int, str]]],
    stats: ValidationStats,
    per_spot: defaultdict[str, PerSpotMetricsAccumulator],
) -> None:
    """Update aggregate and per-spot accumulators for one inference frame."""
    for sp in spots:
        sid = sp["spot_id"]
        pred = sp["status"]
        spot_acc = per_spot[sid]
        intervals = gt_by_spot.get(sid)
        if intervals is None:
            stats.n_skipped_spot_not_in_gt += 1
            spot_acc.n_skipped_spot_not_in_gt += 1
            continue
        gt_label, skip = gt_lookup(intervals, frame_index)
        if skip == "gt_unknown":
            stats.n_skipped_gt_unknown += 1
            spot_acc.n_skipped_gt_unknown += 1
            continue
        if skip == "no_gt" or gt_label is None:
            stats.n_skipped_no_gt_interval += 1
            spot_acc.n_skipped_no_gt_interval += 1
            continue
        stats.record(
            gt=gt_label,
            pred=pred,
            meta={
                "frame_index": frame_index,
                "inference_index": inference_index,
                "spot_id": sid,
            },
        )
        spot_acc.record_match(gt=gt_label, pred=pred)


def build_metrics_payload(
    *,
    stats: ValidationStats,
    per_spot: defaultdict[str, PerSpotMetricsAccumulator],
    gt_path: Path,
    json_path: Path,
    source: str,
    stride: int,
    infer_count: int,
    frames_read: int,
) -> dict[str, Any]:
    metrics = stats.metrics_dict()
    per_spot_metrics = {
        sid: per_spot[sid].to_metrics_dict()
        for sid in sorted(per_spot.keys(), key=_spot_sort_key)
    }
    return {
        "gt_csv": str(gt_path),
        "json_regions": str(json_path),
        "source": source,
        "stride": stride,
        "inferences": infer_count,
        "frames_read": frames_read,
        **metrics,
        "per_spot": per_spot_metrics,
    }


def write_metrics_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_disagreements_csv(path: Path, disagreements: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as df:
        wcsv = csv.DictWriter(
            df,
            fieldnames=["frame_index", "inference_index", "spot_id", "gt", "pred"],
        )
        wcsv.writeheader()
        for row in disagreements:
            wcsv.writerow(
                {
                    "frame_index": row["frame_index"],
                    "inference_index": row["inference_index"],
                    "spot_id": row["spot_id"],
                    "gt": row["gt"],
                    "pred": row["pred"],
                }
            )


def write_validation_event(
    events_file: TextIO,
    *,
    source: str,
    frame_index: int,
    inference_index: int,
    stride: int,
    spots: list[dict[str, str]],
) -> None:
    ev = {
        "source": source,
        "frame_index": frame_index,
        "inference_index": inference_index,
        "stride": stride,
        "spots": spots,
    }
    events_file.write(json.dumps(ev) + "\n")
    events_file.flush()
