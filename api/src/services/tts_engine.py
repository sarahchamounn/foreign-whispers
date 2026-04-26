import logging as _logging
import os
import pathlib
import json
import glob
import tempfile

import requests
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from foreign_whispers.voice_resolution import resolve_segment_voice

CHATTERBOX_API_URL = os.getenv("CHATTERBOX_API_URL", "http://localhost:8020")
CHATTERBOX_SPEAKER_WAV = os.getenv("CHATTERBOX_SPEAKER_WAV", "")


class ChatterboxClient:
    def __init__(self, base_url: str = CHATTERBOX_API_URL, speaker_wav: str = CHATTERBOX_SPEAKER_WAV):
        self.base_url = base_url.rstrip("/")
        self.speaker_wav = speaker_wav

    def tts_to_file(self, text: str, file_path: str, **kwargs) -> None:
        chunks = self._split_text(text) if len(text) > 200 else [text]
        combined = AudioSegment.empty()

        speaker_wav = kwargs.get("speaker_wav", self.speaker_wav)

        for idx, chunk in enumerate(chunks):
            if speaker_wav:
                wav_bytes = self._synthesize_with_voice(chunk, speaker_wav)
            else:
                wav_bytes = self._synthesize_default(chunk)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                tmp.write(wav_bytes)
                tmp.flush()
                part = AudioSegment.from_wav(tmp.name)

            part = _trim_audio_silence(part)
            combined += part

            if idx < len(chunks) - 1:
                combined += AudioSegment.silent(duration=40)

        combined.export(file_path, format="wav")

    def _synthesize_default(self, text: str) -> bytes:
        resp = requests.post(
            f"{self.base_url}/v1/audio/speech",
            json={"input": text, "response_format": "wav"},
            timeout=(5, 60),
        )
        resp.raise_for_status()
        return resp.content

    def _synthesize_with_voice(self, text: str, speaker_wav: str) -> bytes:
        speakers_base = pathlib.Path(__file__).parent.parent.parent.parent / "pipeline_data" / "speakers"
        wav_path = speakers_base / speaker_wav

        if not wav_path.exists():
            wav_path = pathlib.Path(speaker_wav)

        if not wav_path.exists():
            _logging.getLogger(__name__).warning(
                "[tts] Speaker WAV %s not found, falling back to default voice", speaker_wav
            )
            return self._synthesize_default(text)

        with open(wav_path, "rb") as f:
            resp = requests.post(
                f"{self.base_url}/v1/audio/speech/upload",
                data={"input": text, "response_format": "wav"},
                files={"voice_file": (wav_path.name, f, "audio/wav")},
                timeout=(5, 60),
            )
        resp.raise_for_status()
        return resp.content

    @staticmethod
    def _split_text(text: str, max_len: int = 200) -> list[str]:
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, current = [], ""

        for s in sentences:
            if current and len(current) + len(s) + 1 > max_len:
                chunks.append(current.strip())
                current = s
            else:
                current = f"{current} {s}".strip() if current else s

        if current:
            chunks.append(current.strip())

        return chunks if chunks else [text]


def _make_tts_engine():
    import functools
    import torch
    from TTS.api import TTS as CoquiTTS

    original_torch_load = torch.load

    @functools.wraps(original_torch_load)
    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = patched_load

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[tts] FORCING local Coqui TTS on {device}")

    return CoquiTTS(
        model_name="tts_models/es/mai/tacotron2-DDC",
        progress_bar=False,
    ).to(device)


_tts_engine = None


def _get_tts_engine():
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = _make_tts_engine()
    return _tts_engine


def _trim_audio_silence(
    audio: AudioSegment,
    min_silence_len: int = 150,
    silence_thresh_offset: int = 16,
) -> AudioSegment:
    if len(audio) == 0:
        return audio

    silence_thresh = audio.dBFS - silence_thresh_offset if audio.dBFS != float("-inf") else -40
    nonsilent = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
    )

    if not nonsilent:
        return audio

    start = nonsilent[0][0]
    end = nonsilent[-1][1]
    return audio[start:end]


def text_from_file(file_path) -> str:
    with open(file_path, "r") as file:
        trans = json.load(file)
    return trans.get("text", "")


def segments_from_file(file_path) -> list[dict]:
    with open(file_path, "r") as file:
        trans = json.load(file)
    return trans.get("segments", [])


def files_from_dir(dir_path) -> list:
    suffix = ".json"
    pth = pathlib.Path(dir_path)
    if not pth.exists():
        raise ValueError("provided path does not exist")

    es_files = glob.glob(str(pth / "*.json"))
    if not es_files:
        raise ValueError(f"no {suffix} files found in {pth}")

    return es_files


def text_to_speech(text, output_file_path):
    _get_tts_engine().tts_to_file(text=text, file_path=str(output_file_path))


def text_file_to_speech(
    source_path,
    output_path,
    tts_engine=None,
    *,
    alignment=None,
    speaker_wav=None,
):
    """Reliable Spanish TTS using gTTS, with safe Chatterbox per-speaker voice test."""
    import subprocess
    import sys
    from gtts import gTTS

    save_name = pathlib.Path(source_path).stem + ".wav"
    save_path = pathlib.Path(output_path) / save_name

    print(f"generating {save_name} with gTTS...", end="")

    with open(source_path, "r") as f:
        data = json.load(f)

    segments = data.get("segments", [])

    # Safe test for per-speaker Chatterbox voice selection.
    # This does NOT replace gTTS. It only tests whether speaker-based voices work.
    if segments:
        try:
            client = ChatterboxClient()
            speakers_dir = pathlib.Path("pipeline_data/speakers")

            for seg in segments[:2]:
                text = seg.get("text", "").strip()
                if not text:
                    continue

                voice = resolve_segment_voice(
                    speakers_dir=speakers_dir,
                    target_language="es",
                    segment=seg,
                )

                print(f"\n[tts-test] speaker={seg.get('speaker', 'SPEAKER_00')} voice={voice}")

                if voice:
                    client.tts_to_file(
                        text=text,
                        file_path="test_output.wav",
                        speaker_wav=voice,
                    )
                    print("[tts-test] SUCCESS with per-speaker Chatterbox voice")
                else:
                    print("[tts-test] No speaker voice found, keeping gTTS fallback")

        except Exception as e:
            print(f"\n[tts-test] Chatterbox per-speaker test failed, keeping gTTS fallback: {e}")

    if segments:
        full_text = " ".join(
            seg.get("text", "").strip()
            for seg in segments
            if seg.get("text", "").strip()
        )
    else:
        full_text = data.get("text", "").strip()

    if not full_text:
        raise ValueError(f"No translated text found in {source_path}")

    # Install gTTS inside container if missing
    try:
        import gtts
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gTTS"])
        from gtts import gTTS

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp_mp3:
        tts = gTTS(text=full_text, lang="es")
        tts.save(tmp_mp3.name)

        audio = AudioSegment.from_mp3(tmp_mp3.name)
        audio.export(str(save_path), format="wav")

    print(" success!")
    return None


if __name__ == "__main__":
    SOURCE_PATH = "./data/transcriptions/es"
    OUTPUT_PATH = "./audios/"

    pathlib.Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    files = files_from_dir(SOURCE_PATH)
    for file in files:
        text_file_to_speech(file, OUTPUT_PATH)