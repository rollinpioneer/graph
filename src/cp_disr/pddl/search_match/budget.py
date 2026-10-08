"""Wall-clock / memory / expansion budget of one search run and the node-milestone snapshots (card C1-DEPOTS-SEARCH-MATCH-V1, plan sections 5.3 and 11).

One budget object per problem. The wall clock starts BEFORE the problem is parsed and grounded and covers parsing, grounding / translation, template construction, state conversion, search and scoring. The memory
limit is the growth of the resident set of the worker process since the start of the problem (after a garbage collection), so that a neural worker (which keeps a CUDA context resident) is not handicapped.
"""
from __future__ import annotations

import gc
import os
import signal
import time

MILESTONES = (1_000, 10_000, 100_000)
PAGE = os.sysconf("SC_PAGE_SIZE")


def rss_bytes():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * PAGE


class ResourceStop(Exception):
    """Raised inside the search when a resource limit is reached; ``status`` is TIMEOUT or MEMORY_LIMIT."""

    def __init__(self, status):
        super().__init__(status)
        self.status = status


class HardTimeout(BaseException):
    """Raised by the SIGALRM guard when a non-cooperative section overruns the wall limit by more than the grace period."""


class Budget:
    def __init__(self, wall_seconds=300.0, max_expansions=100_000, max_rss_growth=8 * 2 ** 30, grace_seconds=20.0):
        gc.collect()
        self.wall_seconds, self.max_expansions, self.max_rss_growth, self.grace = wall_seconds, max_expansions, max_rss_growth, grace_seconds
        self.rss0 = rss_bytes()
        self.t0 = time.perf_counter()
        self.peak_rss_growth = 0
        self._tick = 0

    def elapsed(self):
        return time.perf_counter() - self.t0

    def rss_growth(self):
        g = rss_bytes() - self.rss0
        self.peak_rss_growth = max(self.peak_rss_growth, g)
        return g

    def check(self, every=1):
        """Cheap cooperative check; memory is sampled every ``every`` calls."""
        if self.elapsed() > self.wall_seconds:
            raise ResourceStop("TIMEOUT")
        self._tick += 1
        if self._tick % every == 0 and self.rss_growth() > self.max_rss_growth:
            raise ResourceStop("MEMORY_LIMIT")

    def arm_hard_timer(self):
        """SIGALRM guard for sections that never call ``check`` (grounding, translation): fires ``grace`` seconds after the wall limit."""
        def handler(signum, frame):
            raise HardTimeout()
        signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, self.wall_seconds + self.grace)

    @staticmethod
    def disarm_hard_timer():
        signal.setitimer(signal.ITIMER_REAL, 0)
