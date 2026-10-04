"""Frozen text-only semantic prior for ALFWorld receptacle search.

Request content is exactly: target object class + sorted receptacle classes visible in the room
+ the output schema. No instance contents, no facts, no policy/reward/oracle data can enter:
`build_messages` takes only those two public arguments.
One call per distinct (target class, room receptacle-class set); results are cached and frozen.
"""
import json
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from ...common import digest
from .observation import type_of

MODEL = "qwen3.7-flash-2026-07-15"
FALLBACK_MODEL = "qwen3.8-27b"
ENDPOINT = "https://dashscope.aliyuncs.com/api/v1"
SECRET_KEYS = ("api_key", "authorization", "token", "secret", "password")

SYSTEM = "You estimate where everyday household objects are usually found. Reply with one JSON object only."


def prior_key(otype, rtypes):
    return otype + "|" + ",".join(sorted(set(rtypes)))


def build_messages(otype, rtypes):
    rtypes = sorted(set(rtypes))
    user = (
        "Target object class: %s\n"
        "Receptacle classes present in this room: %s\n"
        "For every listed receptacle class, give a score between 0 and 1 for how likely an instance of the "
        "target object class is to be found in or on that receptacle class in a typical home.\n"
        'Return exactly one JSON object of the form {"scores": {"<receptacle class>": <number>, ...}} '
        "that covers only the listed classes." % (otype, ", ".join(rtypes))
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_scores(text, rtypes):
    """Return {class: score} for listed classes only; missing classes score 0; None if unusable."""
    try:
        obj = json.loads(text)
        raw = obj["scores"]
        if not isinstance(raw, dict):
            return None
    except Exception:
        return None
    out = {}
    for c in rtypes:
        v = raw.get(c, 0.0)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            v = 0.0
        out[c] = float(v)
    return out if any(out.values()) else None


def instance_scores(class_scores, room_receptacles, feasible, can_contain, otype):
    """Spec 3.5.2: drop classes that cannot contain the target, normalise over feasible classes, split
    each class's mass equally over its feasible instances. All-zero/absent prior gives all-zero scores."""
    feas_classes = sorted({type_of(r) for r in feasible})
    if not class_scores:
        return {r: 0.0 for r in feasible}
    mass = {c: class_scores.get(c, 0.0) for c in feas_classes if can_contain(c, otype)}
    z = sum(mass.values())
    if z <= 0:
        return {r: 0.0 for r in feasible}
    count = defaultdict(int)
    for r in feasible:
        count[type_of(r)] += 1
    return {r: mass.get(type_of(r), 0.0) / z / count[type_of(r)] for r in feasible}


class PriorCache:
    def __init__(self, entries=None, meta=None):
        self.entries = dict(entries or {})
        self.meta = dict(meta or {})

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        return cls(d["entries"], d["meta"])

    def save(self, path):
        json.dump({"meta": self.meta, "entries": self.entries}, open(path, "w"), sort_keys=True, indent=1)

    def hash(self):
        return digest({"meta": self.meta, "entries": self.entries})

    def class_scores(self, otype, room_rtypes):
        e = self.entries.get(prior_key(otype, room_rtypes))
        return None if e is None else e.get("scores")

    def scores(self, pub, tables):
        """Instance-level prior scores for one episode (public arguments only)."""
        room = sorted({type_of(r) for r in pub.receptacles})
        cs = self.class_scores(pub.goal_otype, room)
        return instance_scores(cs, pub.receptacles, pub.feasible, tables.can_contain, pub.goal_otype)

    def class_ranking(self, pub, tables):
        """Feasible classes ordered by prior score (ties by name); empty when the prior is unusable."""
        room = sorted({type_of(r) for r in pub.receptacles})
        cs = self.class_scores(pub.goal_otype, room)
        feas = sorted({type_of(r) for r in pub.feasible})
        if not cs:
            return []
        return sorted(feas, key=lambda c: (-cs.get(c, 0.0), c))


class SyntheticPrior:
    """ENGINEERING ONLY (throughput/unit tests): hash-derived class scores, no provider call.
    Never used for any research result; results produced with it are labelled synthetic."""

    synthetic = True

    def scores(self, pub, tables):
        import hashlib

        cs = {c: 0.05 + (int(hashlib.sha256((pub.goal_otype + c).encode()).hexdigest(), 16) % 1000) / 1000.0
              for c in {type_of(r) for r in pub.receptacles}}
        return instance_scores(cs, pub.receptacles, pub.feasible, tables.can_contain, pub.goal_otype)

    def class_scores(self, otype, room_rtypes):
        import hashlib

        return {c: 0.05 + (int(hashlib.sha256((otype + c).encode()).hexdigest(), 16) % 1000) / 1000.0 for c in room_rtypes}


class DashScopeTextProvider:
    """Single-request text transport. The key is read from the environment by the SDK only."""

    def __init__(self, model=MODEL, timeout=60.0, retries=3):
        self.model, self.timeout, self.retries = model, timeout, retries

    def key_present(self):
        return "DASHSCOPE_API_KEY" in os.environ

    def send(self, messages):
        import dashscope
        from dashscope import Generation

        dashscope.base_http_api_url = ENDPOINT
        last = None
        for attempt in range(self.retries):
            started = time.monotonic()
            try:
                r = Generation.call(
                    model=self.model, messages=messages, result_format="message", temperature=0,
                    enable_thinking=False, enable_search=False, response_format={"type": "json_object"},
                    max_tokens=1024, timeout=self.timeout,
                )
                status = int(getattr(r, "status_code", 0))
                if status == 200:
                    text = r.output.choices[0].message.content
                    usage = dict(r.usage) if getattr(r, "usage", None) else {}
                    return {"ok": True, "text": text, "usage": usage, "request_id": getattr(r, "request_id", None),
                            "model": self.model, "latency": time.monotonic() - started}
                last = {"ok": False, "status": status, "code": str(getattr(r, "code", ""))}
                if status in (400, 401, 403, 404):
                    return last
            except Exception as exc:  # never log exception text: it may carry request headers
                last = {"ok": False, "status": 0, "code": type(exc).__name__}
            time.sleep(2.0 * (attempt + 1))
        return last


def generate_cache(keys, provider, workers=16, out_path=None, max_total_tokens=20_000_000):
    """keys: iterable of (otype, tuple(rtypes)). Calls the provider once per key (concurrent), freezes results."""
    keys = sorted({(o, tuple(sorted(set(r)))) for o, r in keys})
    entries, tokens = {}, 0

    def job(k):
        o, rt = k
        res = provider.send(build_messages(o, rt))
        return k, res

    with ThreadPoolExecutor(workers) as ex:
        for k, res in ex.map(job, keys):
            o, rt = k
            e = {"otype": o, "rtypes": list(rt)}
            if res and res.get("ok"):
                e.update(scores=parse_scores(res["text"], list(rt)), raw=res["text"], usage=res.get("usage"),
                         request_id=res.get("request_id"))
                tokens += int((res.get("usage") or {}).get("total_tokens", 0))
            else:
                e.update(scores=None, raw=None, error=res)
            entries[prior_key(o, rt)] = e
            if tokens > max_total_tokens:
                raise RuntimeError("token budget guard tripped")
    cache = PriorCache(entries, {"model": provider.model, "endpoint": ENDPOINT, "n_keys": len(keys), "total_tokens": tokens,
                                 "prompt_hash": digest(build_messages("X", ("a", "b")))})
    if out_path:
        cache.save(out_path)
    return cache
