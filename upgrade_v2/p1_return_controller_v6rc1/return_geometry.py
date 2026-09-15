#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
from typing import Iterable


def norm(v: Iterable[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def quat_normalize(q):
    n = norm(q)
    if n <= 1e-15:
        raise ValueError('zero quaternion')
    return [float(x) / n for x in q]


def quat_conjugate(q):
    q = quat_normalize(q)
    return [q[0], -q[1], -q[2], -q[3]]


def quat_mul_raw(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ]


def quat_rotate(q, v):
    q = quat_normalize(q)
    qv = [0.0, *map(float, v)]
    out = quat_mul_raw(quat_mul_raw(q, qv), quat_conjugate(q))
    return out[1:]


def relative_position_in_eef(p_obj, p_eef, q_eef):
    delta = [float(a) - float(b) for a, b in zip(p_obj, p_eef)]
    return quat_rotate(quat_conjugate(q_eef), delta)


def desired_eef_position(p_obj_checkpoint, q_eef_command, r_eo_regrasp):
    world_offset = quat_rotate(q_eef_command, r_eo_regrasp)
    return [float(a) - float(b) for a, b in zip(p_obj_checkpoint, world_offset)]


def clipnorm(v, max_norm):
    v = [float(x) for x in v]
    n = norm(v)
    if n <= max_norm or n <= 1e-15:
        return v
    scale = float(max_norm) / n
    return [x * scale for x in v]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', required=True)
    args = parser.parse_args()
    payload = json.loads(args.json)
    rel = relative_position_in_eef(
        payload['p_obj_current'], payload['p_eef_current'], payload['q_eef']
    )
    target = desired_eef_position(
        payload['p_obj_checkpoint'], payload['q_eef'], rel
    )
    print(json.dumps({'r_eo': rel, 'p_eef_desired': target}, indent=2))


if __name__ == '__main__':
    main()

COMPENSATION_MODE = 'TRANSLATION_COMPENSATED_FIXED_ORIENTATION'

WORKSPACE = {
    'x': (-1.0, 1.0),
    'y': (-0.8, 0.8),
    'z': (0.35, 1.20),
}

def in_workspace(p):
    x, y, z = (float(v) for v in p)
    return (WORKSPACE['x'][0] <= x <= WORKSPACE['x'][1]
            and WORKSPACE['y'][0] <= y <= WORKSPACE['y'][1]
            and WORKSPACE['z'][0] <= z <= WORKSPACE['z'][1])

def finite_vec(v):
    import math
    out = [float(x) for x in v]
    if any(not math.isfinite(x) for x in out):
        raise ValueError('NaN/Inf')
    return out
