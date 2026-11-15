"""Minimal async chat client for any OpenAI-compatible endpoint (OpenRouter by default),
with on-disk caching so interrupted runs resume, plus a 'mock' backend for dry runs.

Env vars:
  LLM_API_KEY   key for the endpoint (falls back to OPENROUTER_API_KEY)
  LLM_BASE_URL  default https://openrouter.ai/api/v1
"""
import asyncio, hashlib, json, os, random, re
from pathlib import Path
import httpx

_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    for _l in _env.read_text().splitlines():
        if "=" in _l and not _l.startswith("#"):
            _k, _v = _l.split("=", 1); os.environ.setdefault(_k.strip(), _v.strip())

BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
CACHE = Path(__file__).resolve().parent.parent / "results" / "cache.jsonl"

RPM = float(os.environ.get("LLM_RPM", "80"))  # Adaption allows 100 requests/min per key
_next_slot = 0.0
_rate_lock = None
_cache: dict | None = None
_lock = asyncio.Lock()


def _load_cache():
    global _cache
    if _cache is None:
        _cache = {}
        if CACHE.exists():
            for line in CACHE.open():
                try:
                    r = json.loads(line); _cache[r["key"]] = r["text"]
                except Exception:
                    pass
    return _cache


def _key(model, messages, temperature, seed_tag):
    blob = json.dumps([model, messages, temperature, seed_tag], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


async def chat(client, model, messages, temperature=0.7, max_tokens=4000, seed_tag=0, sem=None):
    """Return the assistant text. seed_tag distinguishes repeated samples of the same prompt."""
    cache = _load_cache()
    k = _key(model, messages, temperature, seed_tag)
    if k in cache:
        return cache[k]
    if model.startswith("mock"):
        text, usage = _mock(model, messages, seed_tag), None
    else:
        max_tokens = int(os.environ.get("LLM_MAX_TOKENS", max_tokens))  # override for reasoning reruns; not part of cache key
        text, usage = await _call(client, model, messages, temperature, max_tokens, sem)
    if text.startswith("__ERROR__"):
        return text  # never cache failures
    async with _lock:
        cache[k] = text
        with CACHE.open("a") as f:
            f.write(json.dumps({"key": k, "model": model, "text": text, "usage": usage if not model.startswith("mock") else None}) + "\n")
    return text


async def _call(client, model, messages, temperature, max_tokens, sem):
    name, _, flag = model.partition("@")
    body = {"model": name, "messages": messages, "max_tokens": max_tokens}
    if flag == "noreason":
        body["reasoning_effort"] = "none"
    if temperature is not None:
        body["temperature"] = temperature
    headers = {"Authorization": f"Bearer {API_KEY}", "X-Title": "evidence-selection-eval"}
    global _next_slot, _rate_lock
    if _rate_lock is None:
        _rate_lock = asyncio.Lock()
    for attempt in range(12):
        try:
            async with _rate_lock:  # global pacing: at most RPM request starts per minute
                loop = asyncio.get_running_loop(); now = loop.time()
                wait = _next_slot - now
                _next_slot = max(now, _next_slot) + 60.0 / RPM
            if wait > 0:
                await asyncio.sleep(wait)
            async with sem:
                r = await client.post(f"{BASE_URL}/chat/completions", json=body, headers=headers, timeout=180)
            if r.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(str(r.status_code), request=r.request, response=r)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            if not msg.get("content"):
                raise KeyError("empty content (reasoning budget exhausted?)")
            return msg["content"], r.json().get("usage")
        except (httpx.HTTPError, KeyError, json.JSONDecodeError) as e:
            if attempt == 11:
                return f"__ERROR__ {type(e).__name__}: {e}", None
            hdr = getattr(getattr(e, "response", None), "headers", {}) or {}
            ra, reset = hdr.get("retry-after"), hdr.get("x-ratelimit-reset")
            import time as _t
            if reset and reset.isdigit():
                wait = max(1.0, min(90.0, int(reset) - _t.time() + 1))
            elif ra and ra.replace(".", "").isdigit():
                wait = float(ra)
            else:
                wait = min(60, 3 * 2 ** attempt)
            await asyncio.sleep(wait + random.random() * 2)


# ---------------------------------------------------------------- mock model
def _mock(model, messages, seed_tag):
    """A deliberately biased fake advisor, for testing the analysis end to end.
    'mock-biased' prefers hope-consistent snippets and shifts p; 'mock-fair' ignores hope."""
    rng = random.Random(hash((json.dumps(messages), seed_tag)) & 0xFFFFFFFF)
    text = messages[-1]["content"]
    if "Write your reply" in text:  # two-pass, pass 2
        p = re.search(r"probability of ([0-9.]+)", text).group(1)
        return f"Here's my take.\nPROBABILITY: {p}"
    if "Write 8 short evidence snippets" in text:  # snippet generation
        snips = [{"id": i, "stance": "YES" if i < 4 else "NO", "text": f"Mock consideration {i}."} for i in range(8)]
        return json.dumps({"snippets": snips})
    ids = re.findall(r"^\[([A-H])\] \((?:[^)]*)\)?", text, re.M) or re.findall(r"^\[([A-H])\]", text, re.M)
    stance = dict(re.findall(r"<!--([A-H]):(YES|NO)-->", text))
    low = text.lower(); hope = 0
    if "friend" not in low:
        hope = 1 if "hope this happens" in low else -1 if "hope it doesn't happen" in low else 0
    bias = 0.0 if model == "mock-fair" else 0.6
    scored = []
    for i in ids:
        s = rng.random()
        if hope and stance.get(i) == ("YES" if hope > 0 else "NO"):
            s += bias
        scored.append((s, i))
    sel = [i for _, i in sorted(scored, reverse=True)[:3]]
    nyes = sum(stance.get(i) == "YES" for i in sel)
    p = min(0.97, max(0.03, 0.2 + 0.2 * nyes + 0.05 * hope * (bias > 0) + rng.gauss(0, 0.04)))
    return f"SELECTED: {', '.join(sel)}\nPROBABILITY: {p:.2f}"
