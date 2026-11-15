"""Decide which moments are worth posting, and write the text for each."""
from __future__ import annotations

import re

from .llm import LLM
from .util import log, ts

SCAN_CHUNK = 3600  # seconds of transcript per scan request


def scan_transcript(lines: list[dict], llm: LLM, game: str) -> list[dict]:
    """Read the whole-stream transcript and point at moments the audio/chat
    signals could miss (a funny line said quietly, a story, a big claim)."""
    found: list[dict] = []
    if not lines:
        return found
    end = lines[-1]["e"]
    start = 0.0
    while start < end:
        part = [l for l in lines if start <= l["s"] < start + SCAN_CHUNK]
        start += SCAN_CHUNK
        if len(part) < 5:
            continue
        body = "\n".join(f"[{int(l['s'])}] {l['text']}" for l in part)
        prompt = f"""This is part of the transcript of a live stream{f' of {game}' if game else ''}.
Each line starts with its time in seconds.

Find up to 4 moments that would make a strong short clip on their own: shock, a big
surprise, a fail or a win, real excitement, something funny, an argument, a wild claim.
Only pick moments where the words show that something actually happened. If nothing
stands out, return an empty list.

Reply with JSON only: {{"moments": [{{"t": <seconds where the key moment is>, "why": "<one short line>"}}]}}

TRANSCRIPT
{body}"""
        try:
            for m in llm.ask_json(prompt).get("moments", []):
                found.append({"t": float(m["t"]), "why": str(m.get("why", ""))})
        except Exception as e:  # noqa: BLE001 - scanning is a bonus, never fatal
            log(f"transcript scan skipped for one part: {e}")
    return found


def _block(c: dict) -> str:
    lines = "\n".join(f"  [{l['s']:.1f}-{l['e']:.1f}] {l['text']}" for l in c["lines"]) or "  (no speech found)"
    chat = "\n".join(f"  {m}" for m in c.get("chat", [])[:15]) or "  (no chat data)"
    hint = f"\nspotted in transcript: {c['why']}" if c.get("why") else ""
    return (f"### {c['id']}\nwindow: {c['w0']:.1f} to {c['w1']:.1f} s | peak at {c['t']:.1f} s | "
            f"loudness spike {c['audio_z']} | chat spike {c['chat_z']}{hint}\nspeech:\n{lines}\nchat near the peak:\n{chat}")


def pick(cands: list[dict], llm: LLM, n: int, lo: int, hi: int, ctx: dict, sheets: list) -> list[dict]:
    if llm.enabled:
        try:
            return _pick_llm(cands, llm, n, lo, hi, ctx, sheets)
        except Exception as e:  # noqa: BLE001
            log(f"LLM judging failed ({e}); using simple rules instead")
    return _pick_rules(cands, n, lo, hi, ctx)


def _pick_llm(cands, llm, n, lo, hi, ctx, sheets) -> list[dict]:
    who = ctx.get("streamer") or "the streamer"
    game = ctx.get("game") or "the game"
    vision = ("Each moment also has an image: four frames from around its peak, in time order. "
              "Use them to see what actually happened on screen.\n" if sheets else "")
    prompt = f"""You are choosing clips for YouTube Shorts from a live stream.
Streamer: {who}. Game: {game}. Stream title: {ctx.get('title', '')}.

Below are {len(cands)} candidate moments. They were found by loudness spikes, chat spikes
and a transcript scan, so some are false alarms (a loud ad, a cough, music).
{vision}
Pick the best {n} at most. Fewer is fine: only keep moments a stranger would enjoy
with no context. Good clips have a clear event: shock, surprise, a fail, a clutch win,
real excitement, something funny. Skip anything dull, confusing or cut off.

For each clip you keep:
- start and end, in seconds, inside that moment's window. Length {lo} to {hi} seconds.
  Start at most 3 seconds before the build-up begins, so the viewer is hooked at once.
  End right after the payoff and the reaction. Do not cut a sentence in half.
- score 0-100 for how strong it is.
- reason: one line on why it works.
- hook: on-screen text, at most 6 words, that tells a viewer what is about to happen.
- context: one plain sentence that explains the moment to someone who missed the stream.
- title: at most 70 characters. Must be true to what happens. No all caps.
- description: 1-2 sentences.
- hashtags: 4 to 6 tags without the # sign, lower case, the game and the streamer among them.

Reply with JSON only:
{{"clips": [{{"id": "...", "start": 0.0, "end": 0.0, "score": 0, "reason": "", "hook": "",
"context": "", "title": "", "description": "", "hashtags": []}}]}}

CANDIDATES
{chr(10).join(_block(c) for c in cands)}"""
    data = llm.ask_json(prompt, sheets)
    by_id = {c["id"]: c for c in cands}
    out = []
    for r in data.get("clips", []):
        c = by_id.get(str(r.get("id")))
        if not c:
            continue
        s, e = _fit(float(r["start"]), float(r["end"]), c, lo, hi)
        out.append({**c, "start": s, "end": e, "rank_score": float(r.get("score", 0)),
                    "reason": r.get("reason", ""), "hook": r.get("hook", ""), "context": r.get("context", ""),
                    "title": r.get("title", ""), "description": r.get("description", ""),
                    "hashtags": [str(h).lstrip("#") for h in r.get("hashtags", [])]})
    out.sort(key=lambda x: x["rank_score"], reverse=True)
    return out[:n]


def _fit(s: float, e: float, c: dict, lo: int, hi: int) -> tuple[float, float]:
    s = min(max(s, c["w0"]), c["w1"] - lo)
    e = min(max(e, s + lo), c["w1"], s + hi)
    return round(s, 2), round(e, 2)


def _pick_rules(cands, n, lo, hi, ctx) -> list[dict]:
    """No LLM: keep the strongest spikes and write plain text for them."""
    who = ctx.get("streamer") or "Streamer"
    game = ctx.get("game") or ""
    out = []
    for c in sorted(cands, key=lambda x: x["score"], reverse=True)[:n]:
        s, e = c["t"] - 18, c["t"] + 10
        for l in c["lines"]:  # do not start or stop in the middle of a sentence
            if abs(l["s"] - s) < 3:
                s = l["s"] - 0.2
            if abs(l["e"] - e) < 3:
                e = l["e"] + 0.3
        s, e = _fit(s, e, c, lo, hi)
        near = [l["text"] for l in c["lines"] if c["t"] - 6 <= l["s"] <= c["t"] + 4]
        quote = re.sub(r"\s+", " ", max(near, key=len)) if near else ""
        title = (f'{who}: "{quote[:48]}"' if quote else f"{who} loses it{f' in {game}' if game else ''}")[:70]
        tags = [t for t in [re.sub(r"[^a-z0-9]", "", game.lower()), re.sub(r"[^a-z0-9]", "", who.lower()),
                            "gaming", "streamer", "clips"] if t]
        out.append({**c, "start": s, "end": e, "rank_score": round(c["score"], 1),
                    "reason": f"loudness spike {c['audio_z']}, chat spike {c['chat_z']} at {ts(c['t'])}",
                    "hook": "Wait for it", "context": "", "title": title,
                    "description": f"A moment from {who}'s {game + ' ' if game else ''}stream.", "hashtags": tags})
    return out
