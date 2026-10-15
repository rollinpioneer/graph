"""MuJoCo RGB renderer with fixed front and side camera definitions."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image


class TabletopRenderer:
    def __init__(self, model: object, width: int = 192, height: int = 144) -> None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco

        self._mujoco = mujoco
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._model = model

    def render(self, data: object, view: str, jitter: float = 0.0) -> np.ndarray:
        camera = self._mujoco.MjvCamera()
        self._mujoco.mjv_defaultFreeCamera(self._model, camera)
        camera.type = self._mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (0.0, 0.02, 0.58)
        camera.distance = 2.15
        camera.elevation = -27.0 + jitter
        camera.azimuth = (90.0 if view == "front" else 25.0) + jitter
        self._renderer.update_scene(data, camera=camera)
        return self._renderer.render().copy()

    @staticmethod
    def save_jpeg(array: np.ndarray, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array.astype(np.uint8), mode="RGB").save(path, format="JPEG", quality=86, optimize=True, exif=b"")

    def close(self) -> None:
        self._renderer.close()

    def __enter__(self) -> "TabletopRenderer":
        return self

    def __exit__(self, *unused: object) -> None:
        self.close()
