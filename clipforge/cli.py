"""clipforge: stream link in, ready-to-post Shorts out."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

from . import facecam, judge, render, review, signals
from .llm import LLM
from .source import Source
from .transcribe import Transcriber
from .util import log, need, probe, slug, ts

BEFORE, AFTER = 35, 25  # seconds kept around each candidate peak


def parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="clipforge", description="Find the best moments of a stream and turn them "
                                "into vertical Shorts with captions, a title and a description.")
    p.add_argument("source", help="stream link (Twitch, Kick, YouTube, ...) or a local video file")
    p.add_argument("-n", "--top", type=int, default=6, help="how many clips to make (default 6)")
    p.add_argument("-o", "--out", default="out", help="output folder (default ./out)")
    p.add_argument("--streamer", default="", help="streamer name (default: taken from the link)")
    p.add_argument("--game", default="", help="game name, used in titles and tags")
    p.add_argument("--min", type=int, default=15, dest="lo", help="shortest clip in seconds (default 15)")
    p.add_argument("--max", type=int, default=50, dest="hi", help="longest clip in seconds (default 50)")
    p.add_argument("--llm", default="auto", choices=["auto", "claude-cli", "api", "none"],
                   help="who judges the moments: Claude Code login, Anthropic API, or simple rules")
    p.add_argument("--model", default=None, help="model name for the LLM backend")
    p.add_argument("--no-vision", action="store_true", help="do not show video frames to the LLM")
    p.add_argument("--layout", default="auto", choices=["auto", "stack", "full"],
                   help="stack = face on top, gameplay below; full = whole picture on a blurred fill")
    p.add_argument("--facecam", default="", help="face cam box as x,y,w,h in pixels, if auto-detect misses it")
    p.add_argument("--gameplay", default="crop", choices=["crop", "fit"],
                   help="stack layout: crop the centre of the game (default) or fit the whole frame")
    p.add_argument("--whisper", default="small", help="speech model: tiny, base, small, medium (default small)")
    p.add_argument("--caption-model", default="turbo",
                   help="stronger speech model used only for the captions of the final clips "
                        "(default turbo; 'none' reuses the first pass)")
    p.add_argument("--keep-work", action="store_true",
                   help="keep the downloaded audio and video sections (default: delete them when done)")
    p.add_argument("--language", default=None, help="spoken language code, e.g. en or hi (default: auto)")
    p.add_argument("--no-transcribe", action="store_true", help="skip speech to text (no captions)")
    p.add_argument("--full-transcript", action="store_true",
                   help="transcribe the whole stream so the LLM can scan every word (slow on CPU)")
    p.add_argument("--no-chat", action="store_true", help="do not fetch the chat replay")
    p.add_argument("--credit", action="store_true", help="add a credit line with the stream link to descriptions")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse(argv)
    need("ffmpeg")
    need("ffprobe")
    out_root = Path(a.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. light pass: audio, chat, captions
    src = Source(a.source, want_chat=not a.no_chat)
    src.identify()
    out = out_root / slug(src.title)
    work = out / "_work"
    src.prepare(work)
    streamer = (a.streamer or src.streamer).strip()
    log(f"stream: {src.title} ({ts(src.duration)})")

    llm = LLM(a.llm, a.model, cwd=work)
    log(f"judge: {llm.backend}")
    stt = None if a.no_transcribe else Transcriber(a.whisper, a.language)

    # 2. find candidate moments
    log("measuring loudness")
    db = signals.loudness(src.audio)
    chat = signals.chat_rate(src.chat, len(db)) if src.chat else None
    total, az, cz = signals.score(db, chat)

    extra = []
    transcript = src.captions
    if a.full_transcript and stt and not transcript:
        log("transcribing the whole stream (this is the slow option)")
        transcript = stt.full(src.audio, src.duration or len(db))
    if transcript and llm.enabled:
        log("scanning the full transcript for moments")
        extra = judge.scan_transcript(transcript, llm, a.game)
        log(f"transcript scan found {len(extra)} moments")

    k = a.top * 2 + 4
    cands = signals.candidates(total, az, cz, k, extra=extra)
    if not cands:
        log("no stand-out moments found in this video")
        return 1
    log(f"{len(cands)} candidate moments; fetching video for each")

    # 3. video, speech and frames for each candidate
    sheets, ready = [], []
    for i, c in enumerate(cands):
        c["id"] = f"m{i + 1:02d}"
        c["w0"] = max(0.0, c["t"] - BEFORE)
        c["w1"] = min(src.duration or len(db), c["t"] + AFTER)
        if c["w1"] - c["w0"] < a.lo:
            continue
        try:
            # the cache name carries the times, so a kept section is never reused for another moment
            c["video"], c["offset"] = src.segment(c["w0"], c["w1"], f"sec_{int(c['w0'])}_{int(c['w1'])}")
        except Exception as e:  # noqa: BLE001 - one bad section should not stop the run
            log(f"{c['id']} at {ts(c['t'])}: video fetch failed, skipped ({e})")
            continue
        tr = stt.window(src.audio, c["w0"], c["w1"]) if stt else {"lines": [], "words": []}
        c["lines"], c["words"] = tr["lines"], tr["words"]
        c["chat"] = [m for t, m in src.chat if c["t"] - 3 <= t <= c["t"] + 12 and m]
        pos = c["offset"] + (c["t"] - c["w0"])
        if llm.enabled and not a.no_vision:
            sheet = facecam.contact_sheet(c["video"], [pos - 6, pos - 2, pos + 1, pos + 5], work / f"{c['id']}.jpg")
            if sheet:
                sheets.append(sheet)
        ready.append(c)
        log(f"{c['id']} at {ts(c['t'])}: ready ({len(c['words'])} words)")
    if not ready:
        log("no candidate could be prepared")
        return 1

    # 4. choose the best and write the text
    log("judging the moments")
    clips = judge.pick(ready, llm, a.top, a.lo, a.hi,
                       {"streamer": streamer, "game": a.game, "title": src.title}, sheets)
    if not clips:
        log("the judge kept no clips from this stream")
        return 1

    # 5. layout
    shape = probe(clips[0]["video"])
    vertical = shape["height"] >= 1.4 * shape["width"]
    if vertical:
        log("the stream is already vertical; keeping its own layout")
    fixed = tuple(int(v) for v in a.facecam.split(",")) if a.facecam else None

    captioner = None
    if stt and a.caption_model and a.caption_model != "none":
        if a.caption_model == a.whisper:
            captioner = stt
        else:
            captioner = Transcriber(a.caption_model, stt.language)
            stt._model = None  # free the first model before the bigger one loads
            gc.collect()

    # 6. render (clips from an earlier run of the same stream are replaced)
    for old in out.iterdir():
        if old.is_dir() and old.name[:2].isdigit() and old.name[2:3] == "_":
            shutil.rmtree(old, ignore_errors=True)
    made = []
    for i, c in enumerate(clips, 1):
        d = out / f"{i:02d}_{slug(c['title'], 40)}"
        d.mkdir(parents=True, exist_ok=True)
        dur = c["end"] - c["start"]
        log(f"clip {i}/{len(clips)}: {c['title']}")
        face = fixed
        if not vertical and not fixed and a.layout != "full":
            # the cam can sit in a different place in each part of a stream, so look per clip
            t0 = c["offset"] + (c["start"] - c["w0"])
            face = facecam.detect([facecam.grab(c["video"], t0 + dur * (j + 0.5) / 10) for j in range(10)])
            if face is None:
                log("  no face cam found; using the full-picture layout (pass --facecam x,y,w,h to set it)")
        words = [{"w": w["w"], "s": w["s"] - c["start"], "e": w["e"] - c["start"]}
                 for w in c["words"] if c["start"] <= w["s"] and w["e"] <= c["end"] + 0.3]
        if captioner:  # second, more careful listen for the words shown on screen
            try:
                words = captioner.clip(c["video"], c["offset"] + (c["start"] - c["w0"]), dur)
            except Exception as e:  # noqa: BLE001 - keep the first-pass words
                log(f"  caption model failed ({e}); using the first-pass words")
                captioner = None
        render.render(c["video"], c["offset"] + (c["start"] - c["w0"]), dur, d, words, c["hook"], face, a.gameplay)
        level = render.audio_level(d / "short.mp4")
        if level is None:
            log(f"  WARNING: clip {i} has no sound track")
        elif level < -45:
            log(f"  WARNING: clip {i} is almost silent ({level:.0f} dB)")
        tags = " ".join("#" + t for t in dict.fromkeys(["shorts"] + c["hashtags"]) if t)
        desc = c["description"].strip()
        if a.credit and (streamer or src.url):
            desc += f"\n\nClip from {streamer or 'the'} stream: {src.url}".rstrip(": ")
        desc += f"\n\n{tags}"
        meta = {"title": c["title"][:100], "post_description": desc, "hook": c["hook"], "context": c["context"],
                "hashtags": c["hashtags"], "rank_score": c["rank_score"], "reason": c["reason"],
                "start": c["start"], "end": c["end"], "at": ts(c["start"]), "dir": d.name,
                "streamer": streamer, "game": a.game, "source": src.url or str(a.source),
                "layout": "vertical" if vertical else "stack" if face else "full", "audio_db": level}
        (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        (d / "post.txt").write_text(f"TITLE\n{meta['title']}\n\nDESCRIPTION\n{desc}\n", encoding="utf-8")
        made.append(meta)

    (out / "clips.json").write_text(json.dumps(made, indent=2, ensure_ascii=False), encoding="utf-8")
    page = review.write(out, src.title, made)
    if not a.keep_work and not src.is_local:
        shutil.rmtree(work, ignore_errors=True)
        log("deleted the downloaded audio and video sections")
    elif not a.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    log(f"done: {len(made)} clips in {out}")
    log(f"review page: {page}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
