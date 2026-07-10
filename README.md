# Video Clone — Colab remote worker

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daokimluc/video-clone-colab/blob/main/video_clone_colab_backend.ipynb)

**Open in Colab** (tự load notebook từ repo này):

https://colab.research.google.com/github/daokimluc/video-clone-colab/blob/main/video_clone_colab_backend.ipynb

## Mục đích

Chạy **worker inference** (FastAPI) trên GPU Colab, expose qua Cloudflare Tunnel.  
App desktop **Video Clone** vẫn là control plane (projects / jobs / SQLite) trên máy local.

## Cách dùng

1. Bấm badge **Open In Colab** ở trên (hoặc link).
2. **Runtime → Run all** (hoặc chạy từng cell):
   - Cài deps
   - Tạo `SHARED_SECRET`
   - Start worker `http://127.0.0.1:8765`
   - Tunnel: `cloudflared tunnel --url http://127.0.0.1:8765`
3. Trong app Video Clone → **Cấu hình**:
   - Backend mode = **Remote**
   - Remote URL = URL `https://….trycloudflare.com`
   - Shared secret = `SHARED_SECRET` từ notebook
4. App probe: `GET {remote_url}/health` với header `X-VC-Secret`.

## File

| File | Mô tả |
|------|--------|
| `video_clone_colab_backend.ipynb` | Notebook worker + hướng dẫn tunnel |

## Liên kết

- Desktop app (monorepo): Video Clone  
- Mirror local (trong monorepo): `notebooks/video_clone_colab_backend.ipynb`
