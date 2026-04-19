from pydantic import BaseModel

class SpeakerSegment(BaseModel):
    start_s: float
    end_s: float
    speaker: str


class DiarizeResponse(BaseModel):
    video_id: str
    diarization_path: str
    segments: list[SpeakerSegment]