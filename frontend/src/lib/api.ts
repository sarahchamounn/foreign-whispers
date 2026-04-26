import type {
  DownloadResponse,
  TranscribeResponse,
  TranslateResponse,
  TTSResponse,
  StitchResponse,
  DiarizeResponse,
} from "./types";

const API_BASE =
  typeof window === "undefined"
    ? (process.env.API_URL || "http://api:8080")
    : (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080");

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options?.headers },
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "Unknown error");
    throw new ApiError(res.status, text);
  }

  return res.json();
}

export async function downloadVideo(videoId: string): Promise<DownloadResponse> {
  return fetchJson<DownloadResponse>(`/api/download/${videoId}`, {
    method: "POST",
  });
}

export async function transcribeVideo(
  videoId: string,
  useYoutubeCaptions = true
): Promise<TranscribeResponse> {
  const params = useYoutubeCaptions ? "" : "?use_youtube_captions=false";
  return fetchJson<TranscribeResponse>(`/api/transcribe/${videoId}${params}`, {
    method: "POST",
  });
}

export async function translateVideo(
  videoId: string,
  targetLanguage = "es"
): Promise<TranslateResponse> {
  return fetchJson<TranslateResponse>(
    `/api/translate/${videoId}?target_language=${targetLanguage}`,
    { method: "POST" }
  );
}

export async function synthesizeSpeech(
  videoId: string,
  config: string,
  alignment: boolean = false
): Promise<TTSResponse> {
  return fetchJson<TTSResponse>(
    `/api/tts/${videoId}?config=${config}&alignment=${alignment}`,
    { method: "POST" }
  );
}

export async function stitchVideo(
  videoId: string,
  config: string
): Promise<StitchResponse> {
  return fetchJson<StitchResponse>(
    `/api/stitch/${videoId}?config=${config}`,
    { method: "POST" }
  );
}

export async function diarizeVideo(videoId: string): Promise<DiarizeResponse> {
  return fetchJson<DiarizeResponse>(`/api/diarize/${videoId}`, {
    method: "POST",
  });
}

export function getVideoUrl(videoId: string, config: string): string {
  return `${API_BASE}/api/video/${videoId}?config=${config}`;
}

export function getOriginalVideoUrl(videoId: string): string {
  return `${API_BASE}/api/video/${videoId}/original`;
}

export function getAudioUrl(videoId: string, config: string): string {
  return `${API_BASE}/api/audio/${videoId}?config=${config}`;
}

export function getCaptionsUrl(videoId: string): string {
  return `${API_BASE}/api/captions/${videoId}`;
}

export function getOriginalCaptionsUrl(videoId: string): string {
  return `${API_BASE}/api/captions/${videoId}/original`;
}