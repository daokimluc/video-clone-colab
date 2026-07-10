# Video Clone — Colab remote worker

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daokimluc/video-clone-colab/blob/main/video_clone_colab_backend.ipynb)

**Open in Colab** (tự load notebook từ repo này):

https://colab.research.google.com/github/daokimluc/video-clone-colab/blob/main/video_clone_colab_backend.ipynb

## Mục đích

Chạy **worker FastAPI** trên Colab + **tự tạo Cloudflare quick tunnel** và in ra:

- `REMOTE_URL` = `https://….trycloudflare.com`
- `SHARED_SECRET`

Dán 2 giá trị đó vào desktop **Video Clone → Cấu hình → Remote**.

## Cách dùng

1. Bấm **Open In Colab** ở trên  
2. **Runtime → Run all**  
3. Đợi cell **Tunnel** in `REMOTE_URL` + `SHARED_SECRET`  
4. App: mode **Remote**, dán URL + secret → Lưu  
5. Readiness probe: `GET {REMOTE_URL}/health` header `X-VC-Secret`

**Giữ tab Colab / runtime đang chạy** — disconnect = tunnel chết.

## File

| File | Mô tả |
|------|--------|
| `video_clone_colab_backend.ipynb` | Install → secret → worker :8765 → cloudflared auto URL |

## Lưu ý

- Desktop vẫn là control plane (projects/jobs/SQLite).  
- `/infer/asr` trong notebook hiện **stub** — job ASR local cho đến khi wire offload.  
- Tunnel URL **đổi mỗi lần** restart cloudflared / runtime.
