"""Deterministic failure analysis and translation re-ranking stubs.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics. The translation re-ranking function proposes shorter
Spanish candidates that better fit a target duration budget.
"""

import dataclasses
import logging
import re

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
        estimated_duration_s: Rough speech-duration estimate.
        score: Lower is better.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""
    estimated_duration_s: float = 0.0
    score: float = 0.0


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report."""
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    retries = report.get("n_translation_retries", 0)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _estimate_es_duration(text: str) -> float:
    """Very simple Spanish duration estimate."""
    text = _normalize_spaces(text)
    if not text:
        return 0.1
    return max(len(text) / 15.0, 0.1)


def _shorten_rule_based(text: str) -> list[tuple[str, str]]:
    """Generate shorter Spanish candidates with simple rules."""
    text = _normalize_spaces(text)
    candidates: list[tuple[str, str]] = []

    # Original
    candidates.append((text, "baseline"))

    # Remove filler words
    shortened = re.sub(
        r"\b(bueno|pues|entonces|realmente|muy|simplemente)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    shortened = _normalize_spaces(shortened)
    if shortened and shortened != text:
        candidates.append((shortened, "removed filler words"))

    # Keep first sentence only
    first_sentence = re.split(r"(?<=[.!?])\s+", text)[0].strip()
    if first_sentence and first_sentence != text:
        candidates.append((first_sentence, "kept first sentence only"))

    # Cut after commas / semicolons / colons
    comma_cut = re.split(r"[,;:]", text)[0].strip()
    if comma_cut and comma_cut != text:
        candidates.append((comma_cut, "cut trailing clause"))

    # Truncate long text
    if len(text) > 80:
        trunc = text[:80].rsplit(" ", 1)[0].strip()
        if trunc and trunc != text:
            candidates.append((trunc, "truncated long text"))

    # Deduplicate while preserving order
    seen = set()
    unique_candidates = []
    for cand_text, rationale in candidates:
        if cand_text not in seen:
            seen.add(cand_text)
            unique_candidates.append((cand_text, rationale))

    return unique_candidates


def get_shorter_translations(source_text: str, baseline_es: str, target_duration_s: float):
    """Return shorter translation candidates ranked by duration fit."""
    candidates = _shorten_rule_based(baseline_es)

    scored = []
    for cand_text, rationale in candidates:
        dur = _estimate_es_duration(cand_text)
        fits = dur <= target_duration_s
        score = abs(target_duration_s - dur)
        if fits:
            score -= 1.0

        scored.append(
            TranslationCandidate(
                text=cand_text,
                char_count=len(cand_text),
                brevity_rationale=rationale,
                estimated_duration_s=dur,
                score=score,
            )
        )

    scored.sort(key=lambda x: x.score)
    return scored