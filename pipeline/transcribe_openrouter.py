"""OpenRouter ASR transcription backend -- an optional, low-cost cloud
alternative to transcribe.py's local Whisper path, using NVIDIA's Nemotron
3.5 ASR Streaming Multilingual 0.6B model (hosted by DeepInfra via
OpenRouter: https://openrouter.ai/nvidia/nemotron-3.5-asr-streaming-multilingual-0.6b).

Used automatically whenever OPENROUTER_API_KEY is set (see .env.example) --
same "automatic if the key exists, else fall back" convention transcribe.py
already uses for ANTHROPIC_API_KEY. At ~$0.000003/second of audio (about 1
cent for a 60-minute episode), this is negligible next to local Whisper CPU
time, and runs in seconds rather than minutes per episode.

OpenRouter's transcription endpoint (POST /api/v1/audio/transcriptions,
JSON body with base64 input_audio.data + format) caps each request to a
size/processing budget well under a full podcast episode, so a downloaded
episode is split into CHUNK_SECONDS-long pieces with ffmpeg, transcribed in
parallel, and stitched back together in published order.
"""
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
MODEL = "nvidia/nemotron-3.5-asr-streaming-multilingual-0.6b"
CHUNK_SECONDS = 300  # 5 min/chunk -- comfortably inside the endpoint's per-request size/time budget
MAX_WORKERS = 4
MIN_AUDIO_COVERAGE = 0.85  # same truncated-download guard transcribe.py's Whisper path uses
USER_AGENT = "podcast-monitor/1.0 (+https://github.com/rjre/podcast-monitor)"


def _load_env_file():
    """Tiny stdlib .env loader -- avoids adding python-dotenv as a dependency
    for one variable. Never overrides a real env var that's already set."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


_load_env_file()


def _ffprobe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def _split_audio(path, chunk_seconds=CHUNK_SECONDS):
    """Re-encode to mono 16kHz/64kbps mp3 (small, ASR-friendly, well under
    the endpoint's size limit even after base64 inflation) and segment in
    one ffmpeg pass."""
    tmp_dir = tempfile.mkdtemp(prefix="or-asr-")
    pattern = os.path.join(tmp_dir, "chunk_%04d.mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
         "-ar", "16000", "-ac", "1", "-b:a", "64k",
         "-f", "segment", "-segment_time", str(chunk_seconds), pattern],
        check=True,
    )
    chunks = sorted(
        os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.startswith("chunk_")
    )
    return chunks, tmp_dir


def _transcribe_chunk(chunk_path, max_attempts=3):
    api_key = os.environ["OPENROUTER_API_KEY"]
    with open(chunk_path, "rb") as f:
        data_b64 = base64.b64encode(f.read()).decode("ascii")
    body = json.dumps({
        "model": MODEL,
        "input_audio": {"data": data_b64, "format": "mp3"},
        "language": "en",
    }).encode("utf-8")

    last_exc = None
    for attempt in range(1, max_attempts + 1):
        req = urllib.request.Request(
            API_URL, data=body, method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                result = json.loads(resp.read())
            return result.get("text", "").strip()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            last_exc = RuntimeError(f"HTTP {exc.code}: {detail[:300]}")
        except (urllib.error.URLError, TimeoutError) as exc:
            last_exc = exc
        print(f"      chunk {os.path.basename(chunk_path)} attempt {attempt}/{max_attempts} "
              f"failed ({last_exc}); retrying", file=sys.stderr)
    raise last_exc


class _Info:
    """Minimal stand-in for faster-whisper's TranscriptionInfo, so the rest
    of transcribe.py's pipeline (language field, duration guard) doesn't
    need to branch on which engine produced the transcript."""
    def __init__(self, duration, language="en"):
        self.duration = duration
        self.language = language


def download_and_transcribe(ep, max_attempts=3):
    """Same download-retry/truncation-guard contract as transcribe.py's
    download_and_transcribe(), transcribing via the OpenRouter ASR API
    instead of local Whisper."""
    from transcribe import download_audio, duration_seconds

    expected = duration_seconds(ep.get("duration"))
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        chunk_dir = None
        try:
            download_audio(ep["audio_url"], tmp_path)
            decoded_duration = _ffprobe_duration(tmp_path)
            if expected and decoded_duration < MIN_AUDIO_COVERAGE * expected:
                raise RuntimeError(
                    f"truncated download: decoded {decoded_duration:.0f}s of {expected}s expected"
                )
            chunks, chunk_dir = _split_audio(tmp_path)
            with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(chunks))) as pool:
                texts = list(pool.map(_transcribe_chunk, chunks))
            text = " ".join(t for t in texts if t).strip()
            return text, _Info(decoded_duration)
        except Exception as exc:
            last_exc = exc
            print(f"    attempt {attempt}/{max_attempts} failed ({exc}); retrying", file=sys.stderr)
            continue
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if chunk_dir and os.path.isdir(chunk_dir):
                shutil.rmtree(chunk_dir, ignore_errors=True)
    raise last_exc or RuntimeError("transcription failed after retries")
