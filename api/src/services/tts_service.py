"""HTTP-agnostic service wrapping TTS engine functions."""

import pathlib
from pathlib import Path
from typing import Any

from api.src.services.tts_engine import text_file_to_speech as tts_text_file_to_speech
from foreign_whispers.voice_resolution import resolve_speaker_wav

SPEAKERS_DIR = Path("pipeline_data/speakers")


class TTSService:
    """Thin wrapper around the TTS pipeline."""

    def __init__(self, ui_dir: Path, tts_engine: Any) -> None:
        self.ui_dir = ui_dir
        self.tts_engine = tts_engine

    def text_file_to_speech(
        self,
        source_path: str,
        output_path: str,
        *,
        alignment: bool | None = None,
        speaker_wav: str | None = None,
        target_language: str = "es",
        speaker_id: str | None = None,
    ) -> None:
        """Generate TTS audio from a translated JSON transcript."""
        resolved_wav = None

        tts_text_file_to_speech(
            source_path,
            output_path,
            tts_engine=None,
            alignment=alignment,
            speaker_wav=resolved_wav,
        )

    @staticmethod
    def title_for_video_id(video_id: str, search_dir: pathlib.Path) -> str | None:
        """Find a title by scanning search_dir for JSON files."""
        for f in search_dir.glob("*.json"):
            return f.stem
        return None

    def compute_alignment(
        self,
        en_transcript: dict,
        es_transcript: dict,
        silence_regions: list[dict],
        max_stretch: float = 1.4,
    ) -> list:
        """Run global alignment over EN and ES transcripts."""
        from foreign_whispers.alignment import compute_segment_metrics, global_align
        from foreign_whispers.evaluation import full_evaluation_scorecard

        metrics = compute_segment_metrics(en_transcript, es_transcript)
        aligned = global_align(metrics, silence_regions, max_stretch)

        
        try:
            scorecard = full_evaluation_scorecard(metrics, aligned)

            print("\n=== REAL ALIGNMENT SCORECARD ===")
            for key, value in scorecard.items():
                print(f"{key}: {value}")
        except Exception as e:
            print("Scorecard failed:", e)

        return aligned