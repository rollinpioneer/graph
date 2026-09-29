from pathlib import Path

from cp_disr.platforms.libero import tp_sr_generator as g

ROOT = Path(__file__).resolve().parents[1]
SHA = "0" * 64


def pool():
    return g.generate_pool(SHA, g.production_bounds(ROOT))


def test_G01_pool_size_unique():
    p = pool()
    assert len(p) == 64 and len({c["case_id"] for c in p}) == 64 and len({c["seed"] for c in p}) == 64


def test_G02_sixteen_per_bin():
    p = pool()
    assert all(sum(1 for c in p if c["occlusion_bin"] == b) == 16 for b in range(4))


def test_G03_deterministic():
    assert pool() == pool()


def test_G04_min_separation_and_bin_band():
    for c in pool():
        dx = abs(c["target_xy"][0] - c["second_xy"][0]); dy = abs(c["target_xy"][1] - c["second_xy"][1])
        assert max(dx, dy) >= g.MIN_AXIS_SEPARATION - 1e-9
        lo, hi = g.BIN_BANDS[c["occlusion_bin"]]
        dist = (dx * dx + dy * dy) ** 0.5
        assert lo - 1e-9 <= dist <= hi + 1e-9


def test_G05_container_buffer_ranges_and_lid():
    for c in pool():
        assert g.CONTAINER_X_RANGE[0] <= c["container_xy"][0] <= g.CONTAINER_X_RANGE[1]
        assert g.BUFFER_X_RANGE[0] <= c["buffer_xy"][0] <= g.BUFFER_X_RANGE[1]
        assert c["lid_closed"] is False


def test_G06_split_rows_carry_open_container_and_interferer_role():
    for c in pool():
        r = g.split_row(c)
        assert r["lid_closed"] is False and r["second_role"] == "interferer" and r["split"] == "dev"
