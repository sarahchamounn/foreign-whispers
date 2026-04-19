"""POST /api/diarize/{video_id} — run speaker diarization."""

import json

from fastapi import APIRouter, HTTPException

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.schemas.diarize import DiarizeResponse, SpeakerSegment

router = APIRouter(prefix="/api")


@router.post("/diarize/{video_id}", response_model=DiarizeResponse)
async def diarize_endpoint(video_id: str):
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    transcript_path = settings.transcriptions_dir / f"{title}.json"
    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail="Transcription file not found")

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    diarization_segments = []
    for seg in transcript.get("segments", []):
        diarization_segments.append(
            {
                "start_s": seg.get("start", 0.0),
                "end_s": seg.get("end", 0.0),
                "speaker": "SPEAKER_00",
            }
        )

    settings.diarizations_dir.mkdir(parents=True, exist_ok=True)
    diarization_path = settings.diarizations_dir / f"{title}.json"

    with open(diarization_path, "w", encoding="utf-8") as f:
        json.dump({"segments": diarization_segments}, f, indent=2)

    return DiarizeResponse(
        video_id=video_id,
        diarization_path=str(diarization_path),
        segments=[SpeakerSegment(**seg) for seg in diarization_segments],
    )