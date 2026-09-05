"""stt + tts services — the only GPU-bound part of Sofia.

Both are thin OpenAI-compatible shims (`/v1/audio/transcriptions` and
`/v1/audio/speech`) over PyTorch models. They share one image and one code path
across CUDA and ROCm because PyTorch's HIP build exposes the same `torch.cuda`
API — the Dockerfiles differ only in which wheel index they install from.
"""
