#!/usr/bin/env python3
"""Auto-tune proctoring thresholds from operator labels.

What this script does:
1) Reads labeled events from evidence_folder/labeled_events.jsonl.
2) Optionally reads pose_debug.csv written by newArch.py when POSE_DEBUG=1.
3) Matches labels to the nearest debug frame for the same person_key.
4) Suggests better runtime parameters:
   - POSE_BASE_RADIUS
   - POSE_ANGLE_BASE_RAD
   - POSE_COMBINE_MODE
   - POSE_WARN_THRESHOLD_S
   - POSE_SCORE_K (kept as 1.0 by default; threshold is tuned instead)
5) Writes JSON/CSV outputs you can reuse immediately.

No third-party dependencies are required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LabelRecord:
    ts: float
    decision_ts: float
    decision: str
    label_status: str
    cam_id: str
    track_id: int
    person_key: str
    video_time: str = ""
    real_time: str = ""
    box: list[int] | None = None
    evidence_path: str = ""
    evidence_file: str = ""
    metadata_path: str = ""


@dataclass
class DebugRecord:
    ts: float
    cam_id: str
    person_key: str
    dist: float
    dist_excess: float
    abs_angle: float
    angle_excess_norm: float
    combined_excess: float
    score_s: float


@dataclass
class Sample:
    label: int  # 1=cheating, 0=not_cheating
    label_status: str
    event_ts: float
    decision_ts: float
    cam_id: str
    track_id: int
    person_key: str
    dist: float
    abs_angle: float
    dist_excess: float
    angle_excess_norm: float
    combined_excess: float
    score_s: float


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(float(v) for v in values)
    if p <= 0:
        return vals[0]
    if p >= 100:
        return vals[-1]
    k = (len(vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    d0 = vals[f] * (c - k)
    d1 = vals[c] * (k - f)
    return d0 + d1


def load_labels(path: Path) -> list[LabelRecord]:
    records: list[LabelRecord] = []
    if not path.is_file():
        return records

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            label_status = str(obj.get("label_status") or obj.get("status") or "").strip().lower()
            decision = str(obj.get("decision") or obj.get("operator_decision") or "").strip().lower()
            if label_status not in ("cheating", "not_cheating"):
                if decision in ("cheating", "not_cheating"):
                    label_status = decision
                else:
                    continue

            try:
                ts = float(obj.get("ts", 0.0))
                decision_ts = float(obj.get("decision_ts", obj.get("operator_decision_ts", ts)))
            except Exception:
                continue

            records.append(
                LabelRecord(
                    ts=ts,
                    decision_ts=decision_ts,
                    decision=decision or label_status,
                    label_status=label_status,
                    cam_id=str(obj.get("cam_id", "unknown")),
                    track_id=int(obj.get("track_id", 0) or 0),
                    person_key=str(obj.get("person_key") or f"{obj.get('cam_id', 'unknown')}_{int(obj.get('track_id', 0) or 0)}"),
                    video_time=str(obj.get("video_time", "")),
                    real_time=str(obj.get("real_time", "")),
                    box=obj.get("box") if isinstance(obj.get("box"), list) else None,
                    evidence_path=str(obj.get("evidence_path", "")),
                    evidence_file=str(obj.get("evidence_file", "")),
                    metadata_path=str(obj.get("metadata_path", "")),
                )
            )
    return records


def load_debug(path: Path) -> dict[str, list[DebugRecord]]:
    index: dict[str, list[DebugRecord]] = defaultdict(list)
    if not path.is_file():
        return index

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rec = DebugRecord(
                    ts=float(row.get("ts", 0.0)),
                    cam_id=str(row.get("cid", row.get("cam_id", "unknown"))),
                    person_key=str(row.get("person_key", "")),
                    dist=float(row.get("dist", 0.0)),
                    dist_excess=float(row.get("dist_excess", 0.0)),
                    abs_angle=float(row.get("abs_angle", 0.0)),
                    angle_excess_norm=float(row.get("angle_excess_norm", 0.0)),
                    combined_excess=float(row.get("combined_excess", 0.0)),
                    score_s=float(row.get("score_s", 0.0)),
                )
            except Exception:
                continue
            index[rec.person_key].append(rec)

    for lst in index.values():
        lst.sort(key=lambda r: r.ts)
    return index


def nearest_debug_row(rows: list[DebugRecord], ts: float, max_dt: float = 5.0) -> DebugRecord | None:
    if not rows:
        return None
    best = None
    best_dt = float("inf")
    for rec in rows:
        d = abs(rec.ts - ts)
        if d < best_dt:
            best = rec
            best_dt = d
    if best is None or best_dt > max_dt:
        return None
    return best


def build_samples(labels: list[LabelRecord], debug_index: dict[str, list[DebugRecord]]) -> list[Sample]:
    samples: list[Sample] = []
    for lab in labels:
        rows = debug_index.get(lab.person_key, [])
        rec = nearest_debug_row(rows, lab.ts)
        if rec is None:
            rec = nearest_debug_row(rows, lab.decision_ts)
        if rec is None:
            continue

        label = 1 if lab.label_status == "cheating" else 0
        samples.append(
            Sample(
                label=label,
                label_status=lab.label_status,
                event_ts=lab.ts,
                decision_ts=lab.decision_ts,
                cam_id=lab.cam_id,
                track_id=lab.track_id,
                person_key=lab.person_key,
                dist=rec.dist,
                abs_angle=rec.abs_angle,
                dist_excess=rec.dist_excess,
                angle_excess_norm=rec.angle_excess_norm,
                combined_excess=rec.combined_excess,
                score_s=rec.score_s,
            )
        )
    return samples


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc}


def best_threshold(values: list[float], labels: list[int]) -> tuple[float, dict[str, float]]:
    if not values:
        return 0.0, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    candidates = sorted(set(values))
    # Add midpoints for more stable thresholding.
    candidates += [(a + b) / 2.0 for a, b in zip(candidates, candidates[1:])]
    candidates = sorted(set(candidates))

    best_thr = candidates[0]
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
    best_key = (-1.0, -1.0, -1.0)

    for thr in candidates:
        tp = fp = fn = tn = 0
        for v, y in zip(values, labels):
            pred = 1 if v >= thr else 0
            if pred == 1 and y == 1:
                tp += 1
            elif pred == 1 and y == 0:
                fp += 1
            elif pred == 0 and y == 1:
                fn += 1
            else:
                tn += 1
        metrics = prf(tp, fp, fn)
        key = (metrics["f1"], metrics["recall"], metrics["precision"])
        if key > best_key:
            best_key = key
            best_thr = thr
            best_metrics = metrics

    return best_thr, best_metrics


def evaluate_instant(samples: list[Sample], base_radius: float, angle_base: float, combine_mode: str, angle_weight: float = 1.0) -> tuple[float, dict[str, float]]:
    values: list[float] = []
    labels: list[int] = []
    for s in samples:
        dist_excess = max(0.0, s.dist - base_radius)
        angle_excess = max(0.0, s.abs_angle - angle_base)
        angle_excess_norm = angle_excess / (angle_base + 1e-6)
        if combine_mode == "max":
            score = max(dist_excess, angle_weight * angle_excess_norm)
        else:
            score = dist_excess + angle_weight * angle_excess_norm
        values.append(score)
        labels.append(s.label)
    return best_threshold(values, labels)


def tune(samples: list[Sample]) -> dict[str, Any]:
    if not samples:
        return {
            "ok": False,
            "reason": "No matched samples. Enable POSE_DEBUG=1 for a while, then click Approve/No again.",
        }

    neg = [s for s in samples if s.label == 0]
    pos = [s for s in samples if s.label == 1]
    dist_pool = [s.dist for s in neg] or [s.dist for s in samples]
    angle_pool = [s.abs_angle for s in neg] or [s.abs_angle for s in samples]

    # Candidate grids based on current data distribution.
    base_radius_candidates = sorted(set(
        [percentile(dist_pool, p) for p in (50, 60, 70, 75, 80, 85, 90, 95)]
    ))
    angle_base_candidates = sorted(set(
        [percentile(angle_pool, p) for p in (50, 60, 70, 75, 80, 85, 90, 95)]
    ))

    best = None
    for mode in ("max", "sum"):
        for br in base_radius_candidates:
            for ab in angle_base_candidates:
                thr, metrics = evaluate_instant(samples, br, ab, mode)
                score = (metrics["f1"], metrics["recall"], metrics["precision"])
                candidate = {
                    "POSE_BASE_RADIUS": round(br, 6),
                    "POSE_ANGLE_BASE_RAD": round(ab, 6),
                    "POSE_COMBINE_MODE": mode,
                    "POSE_WARN_THRESHOLD_S": round(thr, 6),
                    "POSE_SCORE_K": 1.0,
                    "metrics": metrics,
                    "instant_threshold": thr,
                }
                if best is None or score > (best["metrics"]["f1"], best["metrics"]["recall"], best["metrics"]["precision"]):
                    best = candidate

    assert best is not None

    # Threshold for accumulated score_s, which is the actual runtime warning threshold.
    score_values = [s.score_s for s in samples]
    score_labels = [s.label for s in samples]
    score_thr, score_metrics = best_threshold(score_values, score_labels)

    # A compact training dataset for future offline training.
    dataset_rows = []
    for s in samples:
        dataset_rows.append({
            "label": s.label,
            "label_status": s.label_status,
            "event_ts": s.event_ts,
            "decision_ts": s.decision_ts,
            "cam_id": s.cam_id,
            "track_id": s.track_id,
            "person_key": s.person_key,
            "dist": s.dist,
            "abs_angle": s.abs_angle,
            "dist_excess": s.dist_excess,
            "angle_excess_norm": s.angle_excess_norm,
            "combined_excess": s.combined_excess,
            "score_s": s.score_s,
        })

    return {
        "ok": True,
        "best_instant": best,
        "best_score_threshold": score_thr,
        "best_score_metrics": score_metrics,
        "matched_samples": len(samples),
        "positive_samples": len(pos),
        "negative_samples": len(neg),
        "dataset_rows": dataset_rows,
        "recommendations": {
            **best,
            "POSE_WARN_THRESHOLD_S": round(score_thr, 6),
            "score_metrics": score_metrics,
        },
    }


def write_outputs(result: dict[str, Any], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "training_result.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    labels_csv = out_dir / "training_dataset.csv"
    rows = result.get("dataset_rows", [])
    if rows:
        fieldnames = list(rows[0].keys())
        with labels_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    env_sh = out_dir / "tuned_env.sh"
    rec = result.get("recommendations", {})
    with env_sh.open("w", encoding="utf-8") as f:
        for key in (
            "POSE_BASE_RADIUS",
            "POSE_ANGLE_BASE_RAD",
            "POSE_COMBINE_MODE",
            "POSE_WARN_THRESHOLD_S",
            "POSE_SCORE_K",
        ):
            if key in rec:
                f.write(f'export {key}="{rec[key]}"\n')

    return {"json": json_path, "csv": labels_csv, "env": env_sh}


def main():
    parser = argparse.ArgumentParser(description="Auto-tune proctoring thresholds from labeled events.")
    parser.add_argument("--labels", type=Path, default=Path("evidence_folder/labeled_events.jsonl"), help="Path to labeled events JSONL.")
    parser.add_argument("--debug", type=Path, default=Path("evidence_folder/pose_debug.csv"), help="Path to pose_debug CSV.")
    parser.add_argument("--out-dir", type=Path, default=Path("evidence_folder/training"), help="Directory for outputs.")
    parser.add_argument("--max-dt", type=float, default=5.0, help="Max timestamp distance for matching labels to debug rows.")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    debug_index = load_debug(args.debug)

    samples = build_samples(labels, debug_index)
    result = tune(samples)

    if not result.get("ok"):
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("\nЧто делать дальше:")
        print("1) Включи POSE_DEBUG=1")
        print("2) Прогони программу и нажми Approve / No несколько раз")
        print("3) Запусти train.py ещё раз")
        return

    outputs = write_outputs(result, args.out_dir)

    print("=== TRAINING SUMMARY ===")
    print(f"labels file: {args.labels}")
    print(f"debug file:  {args.debug}")
    print(f"matched samples: {result['matched_samples']} (pos={result['positive_samples']}, neg={result['negative_samples']})")
    print("\n=== RECOMMENDED ENV ===")
    rec = result["recommendations"]
    for key in ("POSE_BASE_RADIUS", "POSE_ANGLE_BASE_RAD", "POSE_COMBINE_MODE", "POSE_WARN_THRESHOLD_S", "POSE_SCORE_K"):
        print(f"{key}={rec[key]}")
    print("\n=== METRICS ===")
    print(json.dumps({
        "instant_metrics": rec["metrics"],
        "score_metrics": rec["score_metrics"],
    }, ensure_ascii=False, indent=2))
    print("\n=== OUTPUT FILES ===")
    print(f"{outputs['json']}")
    print(f"{outputs['csv']}")
    print(f"{outputs['env']}")


if __name__ == "__main__":
    main()

