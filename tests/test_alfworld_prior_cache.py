"""Prior cache hardening: strict parser, rate limiter, resume, status separation, no provider mixing (no network)."""
import json
import math
import os
import time

import pytest

from cp_disr.platforms.alfworld import prior_provider as P

RT = ("bed", "desk", "drawer")


def test_strict_parser_exact_key_set_and_range():
    ok = '{"scores": {"bed": 0.1, "desk": 1, "drawer": 0.0}}'
    assert P.parse_scores(ok, RT) == ({"bed": 0.1, "desk": 1.0, "drawer": 0.0}, None)
    bad = {
        "not json": "oops",
        "list": "[1]",
        "extra top key": '{"scores": {"bed": 0, "desk": 0, "drawer": 0}, "note": 1}',
        "missing class": '{"scores": {"bed": 0, "desk": 0}}',
        "extra class": '{"scores": {"bed": 0, "desk": 0, "drawer": 0, "sofa": 1}}',
        "out of range high": '{"scores": {"bed": 1.01, "desk": 0, "drawer": 0}}',
        "negative": '{"scores": {"bed": -0.1, "desk": 0, "drawer": 0}}',
        "string": '{"scores": {"bed": "0.5", "desk": 0, "drawer": 0}}',
        "bool": '{"scores": {"bed": true, "desk": 0, "drawer": 0}}',
        "nan": '{"scores": {"bed": NaN, "desk": 0, "drawer": 0}}',
        "inf": '{"scores": {"bed": Infinity, "desk": 0, "drawer": 0}}',
    }
    for name, text in bad.items():
        scores, why = P.parse_scores(text, RT)
        assert scores is None and why, name


class FakeProvider:
    sdk = "9.9.9"

    def __init__(self, model="m-primary", script=None):
        self.model, self.script, self.calls = model, script or {}, []

    def sdk_version(self):
        return self.sdk

    def send(self, messages):
        user = messages[-1]["content"]
        otype = user.split("\n")[0].split(": ")[1]
        self.calls.append(otype)
        kind = self.script.get(otype, "ok")
        if kind == "transport":
            return {"ok": False, "kind": "transport_error", "status": 500, "code": "X"}
        text = {"ok": '{"scores": {"bed": 0.2, "desk": 0.7, "drawer": 0.1}}', "badjson": "nope", "badkeys": '{"scores": {"bed": 0.2}}'}[kind]
        return {"ok": True, "text": text, "usage": {"total_tokens": 10}, "request_id": "rid-" + otype, "model": self.model, "latency": 0.01}


KEYS = [("alarmclock", RT), ("book", RT), ("cd", RT), ("pen", RT)]


def test_status_separation_incremental_save_resume_and_no_provider_mixing(tmp_path):
    out = str(tmp_path / "cache.json")
    prov = FakeProvider(script={"book": "badjson", "cd": "transport"})
    c = P.generate_cache(KEYS, prov, out, rpm=6000, workers=2, save_every=1)
    rep = P.cache_report(c)
    assert (rep["success"], rep["parse_error"], rep["transport_error"]) == (2, 1, 1)
    # parse errors still consumed tokens, hence 30 (three answered requests)
    assert rep["total_tokens"] == 30 and rep["model"] == "m-primary" and rep["sdk_version"] == "9.9.9" and rep["n_request_ids"] == 3
    assert not [f for f in os.listdir(tmp_path) if ".tmp." in f]  # atomic replace leaves no partial file
    on_disk = json.load(open(out))
    assert on_disk["meta"]["prompt_hash"] == P.digest(P.build_messages("X", ("a", "b")))
    # resume: only the transport failure is retried (deterministic parse errors are kept unless asked)
    prov2 = FakeProvider(script={"book": "badjson"})
    c2 = P.generate_cache(KEYS, prov2, out, rpm=6000, resume=True)
    assert prov2.calls == ["cd"]
    assert P.cache_report(c2)["success"] == 3 and P.cache_report(c2)["parse_error"] == 1
    # a different model must never be mixed into this cache
    with pytest.raises(RuntimeError):
        P.generate_cache(KEYS, FakeProvider(model="m-fallback"), out, rpm=6000, resume=True)
    with pytest.raises(RuntimeError):
        P.generate_cache(KEYS, FakeProvider(), out, rpm=6000)  # existing file without --resume


def test_only_successful_entries_serve_scores():
    cache = P.PriorCache({P.prior_key("a", RT): {"status": "parse_error", "scores": None},
                          P.prior_key("b", RT): {"status": "success", "scores": {"bed": 0.1, "desk": 0.2, "drawer": 0.3}}}, {})
    assert cache.class_scores("a", RT) is None and cache.class_scores("b", RT)["drawer"] == 0.3
    assert cache.class_scores("never-asked", RT) is None


def test_rate_limiter_spaces_request_starts():
    lim = P.RateLimiter(rpm=600)  # 0.1 s
    t0 = time.monotonic()
    stamps = []
    for _ in range(6):
        lim.wait()
        stamps.append(time.monotonic())
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert min(gaps) >= 0.09 and stamps[-1] - t0 >= 0.45
    assert math.isclose(P.RateLimiter(55).interval, 60 / 55)
