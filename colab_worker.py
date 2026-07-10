"""
Video Clone — full inference worker for Google Colab (feature parity with local).
Endpoints match desktop remote_worker client:
  GET  /health
  GET  /capabilities
  POST /infer/asr     multipart file + model_size, language, chunk_seconds
  POST /infer/ocr     multipart file + interval, roi_json
  POST /infer/tts     JSON segments → audio_b64 wav items
  POST /infer/translate JSON segments (deep-translator Google)

Run:
  SHARED_SECRET=... uvicorn colab_worker:app --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

SHARED_SECRET = os.environ.get("VC_COLAB_SECRET") or os.environ.get("SHARED_SECRET") or ""
WORKER_PORT = int(os.environ.get("VC_WORKER_PORT", "8765"))

app = FastAPI(title="VideoClone Colab Worker", version="1.0.0")


def check(secret: str | None) -> None:
    expected = SHARED_SECRET or os.environ.get("VC_COLAB_SECRET") or ""
    if not expected:
        raise HTTPException(500, "SHARED_SECRET not set on worker")
    if secret != expected:
        raise HTTPException(401, "bad secret")


def _has_cuda() -> bool:
    # avoid importing torch on every health check if possible
    try:
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            return False
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _pkg_installed(name: str) -> bool:
    """Fast check — do NOT import heavy packages (easyocr/whisper hang first load)."""
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


@app.get("/health")
def health(x_vc_secret: str | None = Header(default=None)) -> dict[str, Any]:
    check(x_vc_secret)
    # Keep health *fast* — no torch/easyocr import here (prevents tunnel timeouts)
    return {
        "ok": True,
        "worker": "colab",
        "port": WORKER_PORT,
        "features": ["asr", "ocr", "tts", "translate"],
    }


@app.get("/capabilities")
def capabilities(x_vc_secret: str | None = Header(default=None)) -> dict[str, Any]:
    check(x_vc_secret)
    # find_spec only — importing easyocr/faster_whisper on first request can block minutes
    whisper = _pkg_installed("faster_whisper")
    easy = _pkg_installed("easyocr")
    edge = _pkg_installed("edge_tts")
    deep = _pkg_installed("deep_translator")
    return {
        "asr": whisper,
        "ocr": easy,
        "tts": edge,
        "translate": deep or True,
        "cuda": _has_cuda(),
        "worker": "colab",
        "parity": {
            "asr": "faster-whisper (GPU if CUDA)",
            "ocr": "easyocr",
            "tts": "edge-tts",
            "translate": "deep-translator Google",
            "export": "local desktop (ffmpeg mix)",
        },
    }


async def _save_upload(upload: UploadFile, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as out:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    if dest.stat().st_size == 0:
        raise HTTPException(400, "empty upload")
    return dest


@app.post("/infer/asr")
async def infer_asr(
    file: UploadFile = File(...),
    model_size: str = Form("small"),
    language: str | None = Form(None),
    chunk_seconds: str | None = Form(None),
    x_vc_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    check(x_vc_secret)
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise HTTPException(500, "faster-whisper not installed on Colab") from exc

    size = (model_size or "small").replace("faster-whisper-", "") or "small"
    lang = language if language and language not in {"auto", "None"} else None
    device = "cuda" if _has_cuda() else "cpu"
    compute = "float16" if device == "cuda" else "int8"

    with tempfile.TemporaryDirectory(prefix="vc-asr-") as td:
        raw = Path(td) / (file.filename or "video.mp4")
        await _save_upload(file, raw)
        wav = Path(td) / "audio.wav"
        # extract 16k mono
        proc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(raw),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(wav),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not wav.is_file():
            raise HTTPException(500, f"ffmpeg extract failed: {(proc.stderr or '')[-800:]}")

        model = WhisperModel(size, device=device, compute_type=compute)
        segments_iter, info = model.transcribe(
            str(wav),
            language=lang,
            vad_filter=True,
            word_timestamps=False,
        )
        segments: list[dict[str, Any]] = []
        for seg in segments_iter:
            text = (seg.text or "").strip()
            if not text:
                continue
            conf = None
            if getattr(seg, "avg_logprob", None) is not None:
                conf = float(min(1.0, max(0.0, 1 + float(seg.avg_logprob))))
            segments.append(
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "source": text,
                    "translation": "",
                    "confidence": conf,
                }
            )

    return {
        "status": "ok",
        "engine": "faster-whisper",
        "device": device,
        "model_size": size,
        "language": getattr(info, "language", lang),
        "segment_count": len(segments),
        "segments": segments,
        "chunk_seconds": chunk_seconds,
    }


@app.post("/infer/ocr")
async def infer_ocr(
    file: UploadFile = File(...),
    interval: str = Form("1.0"),
    roi_json: str | None = Form(None),
    x_vc_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    check(x_vc_secret)
    try:
        import easyocr
    except ImportError as exc:
        raise HTTPException(500, "easyocr not installed — pip install easyocr") from exc

    iv = max(0.3, float(interval or 1.0))
    roi = json.loads(roi_json) if roi_json else {"x": 0, "y": 70, "w": 100, "h": 25, "unit": "percent"}

    with tempfile.TemporaryDirectory(prefix="vc-ocr-") as td:
        raw = Path(td) / (file.filename or "video.mp4")
        await _save_upload(file, raw)
        frames_dir = Path(td) / "frames"
        frames_dir.mkdir()
        # crop ROI percent
        x, y = float(roi.get("x", 0)), float(roi.get("y", 70))
        w, h = float(roi.get("w", 100)), float(roi.get("h", 25))
        unit = str(roi.get("unit") or "percent")
        if unit == "percent":
            vf = (
                f"fps=1/{iv},crop="
                f"iw*{max(w,1)/100:.4f}:ih*{max(h,1)/100:.4f}:"
                f"iw*{max(x,0)/100:.4f}:ih*{max(y,0)/100:.4f}"
            )
        else:
            vf = f"fps=1/{iv},crop={int(w)}:{int(h)}:{int(x)}:{int(y)}"
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", str(raw), "-vf", vf, str(frames_dir / "f_%06d.png")],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise HTTPException(500, f"ffmpeg frames failed: {(proc.stderr or '')[-600:]}")
        frames = sorted(frames_dir.glob("f_*.png"))
        reader = easyocr.Reader(["ch_sim", "en"], gpu=_has_cuda())
        raw_rows: list[tuple[float, str, float]] = []
        for i, fp in enumerate(frames):
            result = reader.readtext(str(fp))
            texts = []
            confs = []
            for item in result or []:
                if len(item) >= 3:
                    texts.append(str(item[1]))
                    confs.append(float(item[2]))
            text = " ".join(texts).strip()
            if text:
                conf = sum(confs) / len(confs) if confs else 0.0
                raw_rows.append((i * iv, text, conf))

        # merge similar consecutive
        segments: list[dict[str, Any]] = []
        if raw_rows:
            cur_s, cur_t, cur_c = raw_rows[0]
            last_t = cur_s
            for t, text, conf in raw_rows[1:]:
                if text.strip().lower() == cur_t.strip().lower() or (
                    len(text) > 3 and text[:8].lower() == cur_t[:8].lower()
                ):
                    last_t = t
                    cur_c = max(cur_c, conf)
                else:
                    segments.append(
                        {
                            "start": cur_s,
                            "end": last_t + iv,
                            "source": cur_t,
                            "translation": "",
                            "confidence": cur_c,
                        }
                    )
                    cur_s, cur_t, cur_c = t, text, conf
                    last_t = t
            segments.append(
                {
                    "start": cur_s,
                    "end": last_t + iv,
                    "source": cur_t,
                    "translation": "",
                    "confidence": cur_c,
                }
            )

    return {"status": "ok", "engine": "easyocr", "segment_count": len(segments), "segments": segments}


@app.post("/infer/tts")
async def infer_tts(
    request: Request,
    x_vc_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    """JSON body (no Pydantic model — works when loaded via importlib on Colab)."""
    check(x_vc_secret)
    try:
        import edge_tts
    except ImportError as exc:
        raise HTTPException(500, "edge-tts not installed") from exc

    body = await request.json()
    default_voice = body.get("default_voice") or "vi-VN-HoaiMyNeural"
    segments = body.get("segments") or []

    items = []
    with tempfile.TemporaryDirectory(prefix="vc-tts-") as td:
        tdir = Path(td)
        for i, seg in enumerate(segments):
            if not isinstance(seg, dict):
                continue
            text = (seg.get("translation") or seg.get("source") or "").strip()
            voice = seg.get("voice") or default_voice
            idx = seg.get("idx") if seg.get("idx") is not None else i + 1
            mp3 = tdir / f"{idx}.mp3"
            wav = tdir / f"{idx}.wav"
            if not text:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        "anullsrc=r=24000:cl=mono",
                        "-t",
                        "0.2",
                        str(wav),
                    ],
                    capture_output=True,
                    check=False,
                )
            else:
                communicate = edge_tts.Communicate(text, voice)
                # endpoint is already async — never asyncio.run() inside FastAPI
                await communicate.save(str(mp3))
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(mp3), str(wav)],
                    capture_output=True,
                    check=False,
                )
            target = max(0.2, float(seg.get("end") or 1) - float(seg.get("start") or 0))
            fitted = tdir / f"{idx}_fit.wav"
            dur = target
            try:
                pr = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "json",
                        str(wav),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                dur = float(json.loads(pr.stdout or "{}").get("format", {}).get("duration") or target)
            except Exception:
                pass
            if dur > 0.05 and abs(dur - target) / max(target, 0.01) > 0.08:
                speed = max(0.5, min(2.0, dur / target))
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(wav),
                        "-filter:a",
                        f"atempo={speed:.4f}",
                        str(fitted),
                    ],
                    capture_output=True,
                    check=False,
                )
                use = fitted if fitted.is_file() else wav
            else:
                use = wav
            if not use.is_file():
                raise HTTPException(500, f"TTS failed for idx={idx}")
            b64 = base64.b64encode(use.read_bytes()).decode("ascii")
            items.append(
                {
                    "idx": idx,
                    "voice": voice,
                    "format": "wav",
                    "audio_b64": b64,
                    "audio_duration": target,
                }
            )
    return {"status": "ok", "engine": "edge-tts", "items": items}


@app.post("/infer/translate")
async def infer_translate(
    request: Request,
    x_vc_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    check(x_vc_secret)
    try:
        from deep_translator import GoogleTranslator
    except ImportError as exc:
        raise HTTPException(500, "deep-translator not installed") from exc

    body = await request.json()
    src = body.get("source_lang") or "auto"
    if src in {"auto", ""}:
        src = "auto"
    tgt = body.get("target_lang") or "vi"
    translator = GoogleTranslator(source=src if src != "auto" else "auto", target=tgt)
    out = []
    for seg in body.get("segments") or []:
        if not isinstance(seg, dict):
            continue
        text = (seg.get("source") or "").strip()
        tr = translator.translate(text) if text else ""
        out.append({"idx": seg.get("idx"), "source": seg.get("source"), "translation": tr})
    return {"status": "ok", "engine": "google", "segments": out}


@app.exception_handler(Exception)
async def _unhandled(request, exc):  # type: ignore[no-untyped-def]
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": str(exc), "trace": traceback.format_exc()[-1500:]},
    )


if __name__ == "__main__":
    import uvicorn

    if not SHARED_SECRET:
        raise SystemExit("Set SHARED_SECRET or VC_COLAB_SECRET")
    uvicorn.run(app, host="127.0.0.1", port=WORKER_PORT, log_level="info")
