import os
import pathlib
import json
import glob
import tempfile
import logging as _logging

import requests
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


CHATTERBOX_API_URL = os.getenv("CHATTERBOX_API_URL", "http://chatterbox-gpu:8020")
CHATTERBOX_SPEAKER_WAV = os.getenv("CHATTERBOX_SPEAKER_WAV", "")

# Keep demo short so Chatterbox does not take forever on CPU
DEMO_SEGMENT_LIMIT = 8


class ChatterboxClient:
    def __init__(self, base_url: str = CHATTERBOX_API_URL, speaker_wav: str = CHATTERBOX_SPEAKER_WAV):
        self.base_url = base_url.rstrip("/")
        self.speaker_wav = speaker_wav

    def tts_to_file(self, text: str, file_path: str, **kwargs) -> None:
        chunks = self._split_text(text) if len(text) > 180 else [text]
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
                combined += AudioSegment.silent(duration=80)

        combined.export(file_path, format="wav")

    def _synthesize_default(self, text: str) -> bytes:
        resp = requests.post(
            f"{self.base_url}/v1/audio/speech",
            json={"input": text, "response_format": "wav"},
            timeout=(10, 180),
        )
        resp.raise_for_status()
        return resp.content

    def _synthesize_with_voice(self, text: str, speaker_wav: str) -> bytes:
        speakers_base = pathlib.Path(__file__).parent.parent.parent.parent / "pipeline_data" / "speakers"
        wav_path = speakers_base / speaker_wav

        if not wav_path.exists():
            wav_path = pathlib.Path(speaker_wav)

        if not wav_path.exists():
            _logging.getLogger(__name__).warning("[tts] Speaker WAV %s not found", speaker_wav)
            raise FileNotFoundError(speaker_wav)

        with open(wav_path, "rb") as f:
            resp = requests.post(
                f"{self.base_url}/v1/audio/speech/upload",
                data={"input": text, "response_format": "wav"},
                files={"voice_file": (wav_path.name, f, "audio/wav")},
                timeout=(10, 240),
            )

        resp.raise_for_status()
        return resp.content

    @staticmethod
    def _split_text(text: str, max_len: int = 180) -> list[str]:
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, current = [], ""

        for sentence in sentences:
            if current and len(current) + len(sentence) + 1 > max_len:
                chunks.append(current.strip())
                current = sentence
            else:
                current = f"{current} {sentence}".strip() if current else sentence

        if current:
            chunks.append(current.strip())

        return chunks if chunks else [text]


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
    client = ChatterboxClient()
    client.tts_to_file(text=text, file_path=str(output_file_path))


def text_file_to_speech(
    source_path,
    output_path,
    tts_engine=None,
    *,
    alignment=None,
    speaker_wav=None,
):
    """
    Generate real translated Spanish TTS using Chatterbox and WAV speaker profiles.

    IMPORTANT:
    - No gTTS.
    - No MP3 fallback.
    - Uses translated text from JSON segments.
    - Uses SPEAKER_00.wav and SPEAKER_01.wav as voice profiles.
    - Outputs WAV.
    - Limits to first DEMO_SEGMENT_LIMIT segments for demo reliability on CPU.
    """
    save_name = pathlib.Path(source_path).stem + ".wav"
    save_path = pathlib.Path(output_path) / save_name

    print(f"generating {save_name}...", flush=True)

    with open(source_path, "r") as f:
        data = json.load(f)

    segments = data.get("segments", [])

    if segments and not any("speaker" in seg for seg in segments):
        try:
            whisper_path = pathlib.Path(
                str(source_path).replace("translations/argos", "transcriptions/whisper")
            )

            if whisper_path.exists():
                with open(whisper_path, "r") as wf:
                    whisper_data = json.load(wf)

                whisper_segments = whisper_data.get("segments", [])

                for seg, wseg in zip(segments, whisper_segments):
                    seg["speaker"] = wseg.get("speaker", "SPEAKER_00")

                print("[tts] Speaker labels injected from Whisper", flush=True)
            else:
                print(f"[tts] Whisper file not found for speaker injection: {whisper_path}", flush=True)

        except Exception as e:
            print(f"[tts] Speaker injection failed: {e}", flush=True)

    if not segments:
        raise ValueError(f"No segments found in {source_path}")

    speaker_00_path = pathlib.Path("/app/pipeline_data/speakers/es/SPEAKER_00.wav")
    speaker_01_path = pathlib.Path("/app/pipeline_data/speakers/es/SPEAKER_01.wav")

    if not speaker_00_path.exists():
        raise FileNotFoundError(f"Missing voice file: {speaker_00_path}")

    if not speaker_01_path.exists():
        raise FileNotFoundError(f"Missing voice file: {speaker_01_path}")

    client = ChatterboxClient()
    combined = AudioSegment.empty()

    demo_segments = segments[:DEMO_SEGMENT_LIMIT]

    for i, seg in enumerate(demo_segments):
        text = seg.get("text", "").strip()
        if not text:
            continue

        speaker = seg.get("speaker","SPEAKER_00")

        if speaker == "SPEAKER_00":
            voice = "es/SPEAKER_00.wav"
        else:
            voice = "es/SPEAKER_01.wav"

        print(
            f"[tts] REAL translated Chatterbox speaker={speaker} voice={voice} text={text[:80]}",
            flush=True,
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav:
            client.tts_to_file(
                text=text,
                file_path=tmp_wav.name,
                speaker_wav=voice,
            )

            audio = AudioSegment.from_wav(tmp_wav.name)
            audio = _trim_audio_silence(audio)

            combined += audio
            combined += AudioSegment.silent(duration=180)

    if len(combined) == 0:
        raise ValueError(f"No audio generated from {source_path}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(save_path), format="wav")

    print("success! (real translated Chatterbox gender TTS demo)", flush=True)
    return None


if __name__ == "__main__":
    SOURCE_PATH = "./data/transcriptions/es"
    OUTPUT_PATH = "./audios/"

    pathlib.Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    files = files_from_dir(SOURCE_PATH)
    for file in files:
        text_file_to_speech(file, OUTPUT_PATH)