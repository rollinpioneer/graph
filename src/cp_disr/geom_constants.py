"""Geometry constants shared by D0 env and TaskEvaluator.

Kept independent of robosuite so production deadline tests can import
TaskEvaluator without collecting the simulator stack.
"""
from __future__ import annotations

import numpy as np

CONTAINER_INNER = np.array([0.030, 0.030])
