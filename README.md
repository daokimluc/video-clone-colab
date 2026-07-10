# Video Clone — Colab full worker

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daokimluc/video-clone-colab/blob/main/video_clone_colab_backend.ipynb)

https://colab.research.google.com/github/daokimluc/video-clone-colab/blob/main/video_clone_colab_backend.ipynb

## Feature parity

| Feature | Local desktop | Colab remote |
|---------|---------------|--------------|
| ASR (Whisper) | CPU/GPU | **GPU Colab** via `/infer/asr` |
| OCR (EasyOCR) | local | **Colab** `/infer/ocr` |
| TTS (edge-tts) | local | **Colab** `/infer/tts` |
| Translate | Google / 9router | Google on Colab *or* local 9router |
| Export MP4 | **local** ffmpeg | — (needs source file) |

## Files

- `colab_worker.py` — FastAPI worker (full infer API)
- `video_clone_colab_backend.ipynb` — install, secret, worker, cloudflared tunnel

## Desktop

1. Open Colab → **Runtime → GPU** → **Run all**
2. Copy `REMOTE_URL` + `SHARED_SECRET`
3. App **Cấu hình → Remote**
4. Run **Nhận dạng** — job message shows `(colab-remote)`
