"""Frozen text-only semantic prior for ALFWorld receptacle search.

Request content is exactly: target object class + sorted receptacle classes visible in the room
+ the output schema. No instance contents, no facts, no policy/reward/oracle data can enter:
`build_messages` takes only those two public arguments.
One call per distinct (target class, room receptacle-class set); results are cached and frozen.
"""
import json
import math
import os
import threading
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
    """Strict parser: {"scores": {...}} whose key set EXACTLY equals the listed classes and whose
    values are finite numbers in [0, 1]. Returns (scores, None) or (None, reason)."""
    try:
        obj = json.loads(text)
    except Exception:
        return None, "invalid_json"
    if not isinstance(obj, dict) or set(obj) != {"scores"} or not isinstance(obj["scores"], dict):
        return None, "bad_envelope"
    raw = obj["scores"]
    if set(raw) != set(rtypes):
        return None, "key_set_mismatch"
    out = {}
    for c in rtypes:
        v = raw[c]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0.0 <= v <= 1.0:
            return None, "bad_value"
        out[c] = float(v)
    return out, None


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
        return None if e is None or e.get("status") != "success" else e.get("scores")

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


class RateLimiter:
    """Spaces request START times by at least 60/rpm seconds across threads."""

    def __init__(self, rpm):
        self.interval, self.next, self.lock = 60.0 / rpm, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next)
            self.next = start + self.interval
        if start > now:
            time.sleep(start - now)


class DashScopeTextProvider:
    """Single-request text transport. The key is read from the environment by the SDK only and is
    never logged, echoed or stored; exceptions are reduced to their type name."""

    def __init__(self, model=MODEL, timeout=60.0, retries=3):
        self.model, self.timeout, self.retries = model, timeout, retries

    def key_present(self):
        return bool(os.environ.get("DASHSCOPE_API_KEY"))

    def sdk_version(self):
        import importlib.metadata

        return importlib.metadata.version("dashscope")

    def send(self, messages):
        """Returns {ok, text, usage, request_id, ...} or {ok False, kind: transport_error, ...}."""
        import dashscope
        from dashscope import MultiModalConversation

        # qwen3.7-flash is served through the multimodal-generation route (text-only content here);
        # the plain text-generation route answers HTTP 400 "url error" for this model.
        dashscope.base_http_api_url = ENDPOINT
        wire = [{"role": m["role"], "content": [{"text": m["content"]}]} for m in messages]
        last = {"ok": False, "kind": "transport_error", "status": 0, "code": "NoAttempt"}
        for attempt in range(self.retries):
            started = time.monotonic()
            try:
                r = MultiModalConversation.call(
                    model=self.model, messages=wire, result_format="message", temperature=0,
                    enable_thinking=False, enable_search=False, response_format={"type": "json_object"},
                    max_tokens=1024, timeout=self.timeout,
                )
                status = int(getattr(r, "status_code", 0))
                if status == 200:
                    usage = dict(r.usage) if getattr(r, "usage", None) else {}
                    content = r.output.choices[0].message.content
                    text = "".join(c.get("text", "") for c in content) if isinstance(content, list) else content
                    return {"ok": True, "text": text, "usage": usage,
                            "request_id": getattr(r, "request_id", None), "model": self.model,
                            "latency": time.monotonic() - started}
                last = {"ok": False, "kind": "transport_error", "status": status, "code": str(getattr(r, "code", "")),
                        "request_id": getattr(r, "request_id", None)}
                if status in (400, 401, 403, 404):
                    return last
            except Exception as exc:  # never log exception text: it may carry request headers
                last = {"ok": False, "kind": "transport_error", "status": 0, "code": type(exc).__name__}
            time.sleep(2.0 * (attempt + 1))
        return last


def atomic_write_json(path, obj):
    tmp = path + ".tmp.%d" % os.getpid()
    with open(tmp, "w") as fh:
        json.dump(obj, fh, sort_keys=True, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def generate_cache(keys, provider, out_path, rpm=55, workers=4, resume=False, retry_parse=False, save_every=10):
    """Rate-limited, resumable, incrementally saved cache. One provider/model per cache file.

    status per key: success | parse_error | transport_error.  keys: iterable of (otype, rtypes)."""
    keys = sorted({(o, tuple(sorted(set(r)))) for o, r in keys})
    prompt_hash = digest(build_messages("X", ("a", "b")))
    meta = {"model": provider.model, "endpoint": ENDPOINT, "route": "multimodal-generation(text-only)", "sdk_version": provider.sdk_version(), "prompt_hash": prompt_hash,
            "rpm_limit": rpm}
    entries = {}
    if resume and os.path.exists(out_path):
        old = json.load(open(out_path))
        for f in ("model", "endpoint", "prompt_hash"):
            if old["meta"].get(f) != meta[f]:
                raise RuntimeError("resume refused: cache was produced with a different %s (no provider mixing)" % f)
        entries = dict(old["entries"])
    elif os.path.exists(out_path):
        raise RuntimeError("output exists; use resume")
    todo = []
    for o, rt in keys:
        e = entries.get(prior_key(o, rt))
        if e is None or e["status"] == "transport_error" or (retry_parse and e["status"] == "parse_error"):
            todo.append((o, rt))
    limiter, lock, done = RateLimiter(rpm), threading.Lock(), [0]

    def totals():
        return sum(int((e.get("usage") or {}).get("total_tokens", 0)) for e in entries.values())

    def save():
        atomic_write_json(out_path, {"meta": {**meta, "total_tokens": totals(), "n_keys": len(keys)}, "entries": entries})

    def job(k):
        o, rt = k
        limiter.wait()
        res = provider.send(build_messages(o, rt))
        e = {"otype": o, "rtypes": list(rt)}
        if res.get("ok"):
            scores, why = parse_scores(res["text"], list(rt))
            e.update(status="success" if scores else "parse_error", scores=scores, parse_error=why, raw=res["text"],
                     usage=res.get("usage"), request_id=res.get("request_id"), model=res.get("model"), latency=res.get("latency"))
        else:
            e.update(status="transport_error", scores=None, error={k2: res.get(k2) for k2 in ("status", "code", "request_id")})
        with lock:
            entries[prior_key(o, rt)] = e
            done[0] += 1
            if done[0] % save_every == 0:
                save()
        return e

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(job, todo))
    save()
    return PriorCache(entries, {**meta, "total_tokens": totals(), "n_keys": len(keys)})


def cache_report(cache, expected_keys=None):
    c = {"success": 0, "parse_error": 0, "transport_error": 0}
    reasons = defaultdict(int)
    for e in cache.entries.values():
        c[e["status"]] += 1
        if e.get("parse_error"):
            reasons[e["parse_error"]] += 1
    n = len(cache.entries)
    return {"n_keys": n, "success": c["success"], "parse_error": c["parse_error"], "transport_error": c["transport_error"],
            "valid_rate": c["success"] / max(1, n), "parse_error_reasons": dict(reasons),
            "total_tokens": cache.meta.get("total_tokens"), "model": cache.meta.get("model"), "sdk_version": cache.meta.get("sdk_version"),
            "prompt_hash": cache.meta.get("prompt_hash"), "models_seen_in_entries": sorted({e.get("model") for e in cache.entries.values() if e.get("model")}),
            "n_request_ids": sum(1 for e in cache.entries.values() if e.get("request_id"))}
