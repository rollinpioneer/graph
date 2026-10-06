"""Deterministic instance generation for the Blocksworld splits (runbook 11). Pure python, hash-seeded, no result is ever an input."""
from __future__ import annotations

import hashlib
import random

from . import state as S

NAMESPACE = "cp_disr_c1_blocksworld_v1"
TRAIN_SHAPES = {3: ((3,), (2, 1)), 4: ((3, 1), (2, 1, 1)), 5: ((3, 1, 1), (2, 1, 1, 1))}
A2_SHAPES = {4: ((2, 2),), 5: ((3, 2), (2, 2, 1))}


def rng_for(*key):
    h = hashlib.sha256(("|".join([NAMESPACE] + [str(k) for k in key])).encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def tower_colors(height, top_color):
    """Colours bottom to top of an alternating tower whose TOP block has ``top_color``."""
    top_to_bottom = [top_color if i % 2 == 0 else 1 - top_color for i in range(height)]
    return list(reversed(top_to_bottom))


def make_colors(n, rng, towers_needed):
    """Random colours with enough blocks of each colour for the alternating goal towers (``towers_needed`` = list of colour sequences)."""
    need = [0, 0]
    for seq in towers_needed:
        for c in seq:
            need[c] += 1
    for _ in range(1000):
        colors = [rng.randint(0, 1) for _ in range(n)]
        if colors.count(S.RED) >= need[S.RED] and colors.count(S.BLUE) >= need[S.BLUE]:
            return colors
    raise RuntimeError("could not draw colours")


def make_goal(n, colors, heights, top_color, rng):
    """Goal configuration: alternating towers of the given heights (top colour fixed) plus singletons on the table. Returns None when the colours cannot support it."""
    pools = {S.RED: [i for i in range(n) if colors[i] == S.RED], S.BLUE: [i for i in range(n) if colors[i] == S.BLUE]}
    for p in pools.values():
        rng.shuffle(p)
    goal = [S.TABLE] * n
    used = set()
    for h in heights:
        if h == 1:
            continue
        seq = tower_colors(h, top_color)
        blocks = []
        for c in seq:
            if not pools[c]:
                return None
            b = pools[c].pop()
            blocks.append(b)
            used.add(b)
        for lo, hi in zip(blocks, blocks[1:]):
            goal[hi] = lo
    return tuple(goal)


def random_initial(n, colors, rng, want_blue_on_red=False, max_tries=500, p_stack=0.55):
    """Random initial configuration (towers on the table, hand empty). Optionally forces a Blue block directly on a Red block. ``p_stack`` controls tower height."""
    for _ in range(max_tries):
        order = list(range(n))
        rng.shuffle(order)
        below = [S.TABLE] * n
        prev = None
        for b in order:
            if prev is not None and rng.random() < p_stack:
                below[b] = prev
            prev = b
        state = tuple(below)
        if not S.is_valid(state):
            continue
        if want_blue_on_red and not blue_on_red(colors, state):
            continue
        return state
    raise RuntimeError("could not draw an initial state")


def blue_on_red(colors, state):
    return any(b >= 0 and colors[x] == S.BLUE and colors[b] == S.RED for x, b in enumerate(state))


def make_problem(split, n, k, heights, top_color, attempt, names=None, want_blue_on_red=False, p_stack=0.55):
    """One instance for (split, n, index k, attempt). ``top_color`` is the colour of the top block of every nontrivial goal tower."""
    rng = rng_for(split, n, k, attempt)
    nontrivial = [h for h in heights if h > 1]
    seqs = [tower_colors(h, top_color) for h in nontrivial]
    colors = make_colors(n, rng, seqs)
    goal = make_goal(n, colors, heights, top_color, rng)
    if goal is None:
        return None
    init = random_initial(n, colors, rng, want_blue_on_red, p_stack=p_stack)
    if init == tuple(goal):
        return None
    return S.Problem(tuple(names or S.default_names(n)), tuple(colors), init, tuple(goal))
