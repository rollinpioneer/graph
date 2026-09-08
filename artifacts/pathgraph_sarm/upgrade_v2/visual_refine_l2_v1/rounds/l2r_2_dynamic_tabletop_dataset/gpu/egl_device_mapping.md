# EGL Device Mapping

- Host inventory: eight physical A100 devices were visible through the approved host `nvidia-smi` query.
- Python CUDA runtime: no CUDA compute device was exposed to this process.
- MuJoCo EGL: one logical EGL device was exposed; valid ID range was `0..0`.
- Rendering device: logical EGL device `0`.
- Physical UUID mapping: not exposed by the current Python/EGL namespace, so no physical UUID is claimed.
- Training or tensor inference: none.
