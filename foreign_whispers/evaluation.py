"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment — no other dependencies.
"""
import statistics as _stats

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    decide_action,
)


def clip_evaluation_report(metrics, aligned):
    if not metrics or not aligned:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch": 0.0,
            "n_gap_shifts": 0,
            "n_translation_retries": 0,
            "total_cumulative_drift_s": 0.0,
        }

    duration_errors = []
    severe_stretch = 0
    gap_shifts = 0
    translation_retries = 0
    total_drift = 0.0

    for i, a in enumerate(aligned):
        stretch = getattr(a, "stretch_factor", 1.0)
        if stretch > 1.25 or stretch < 0.8:
            severe_stretch += 1

        action = getattr(a, "action", "")
        action_val = action.value if hasattr(action, "value") else str(action)

        if "SHIFT" in action_val:
            gap_shifts += 1
        if "SHORTER" in action_val or "RETRY" in action_val:
            translation_retries += 1

        if i < len(metrics):
            m = metrics[i]
            target = getattr(m, "target_duration_s", getattr(m, "duration_s", 0.0))
            pred = getattr(m, "translated_duration_s", getattr(m, "est_duration_s", target))
            duration_errors.append(abs(pred - target))
            total_drift += abs(pred - target)

    n = max(len(aligned), 1)

    return {
        "mean_abs_duration_error_s": sum(duration_errors) / max(len(duration_errors), 1),
        "pct_severe_stretch": 100.0 * severe_stretch / n,
        "n_gap_shifts": gap_shifts,
        "n_translation_retries": translation_retries,
        "total_cumulative_drift_s": total_drift,
    }

def full_evaluation_scorecard(metrics, aligned):
    """
    Creates a simple full evaluation scorecard for Assignment 4.
    This uses the clip-level metrics and gives an overall quality label.
    """
    report = clip_evaluation_report(metrics, aligned)

    mean_error = report["mean_abs_duration_error_s"]
    severe_pct = report["pct_severe_stretch"]
    drift = report["total_cumulative_drift_s"]
    retries = report["n_translation_retries"]
    shifts = report["n_gap_shifts"]

    score = 100

    if mean_error > 1.0:
        score -= 20
    elif mean_error > 0.5:
        score -= 10

    if severe_pct > 30:
        score -= 20
    elif severe_pct > 10:
        score -= 10

    if drift > 10:
        score -= 15
    elif drift > 5:
        score -= 8

    if retries > 3:
        score -= 10

    if shifts > 3:
        score -= 10

    score = max(score, 0)

    if score >= 85:
        quality = "Good"
    elif score >= 70:
        quality = "Acceptable"
    else:
        quality = "Needs improvement"

    return {
        **report,
        "overall_score": score,
        "quality_label": quality,
    }
