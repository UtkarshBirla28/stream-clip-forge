"""Small shared helpers: logging, subprocess, ffprobe, text."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


def log(msg: str) -> None:
    print(f"[clipforge] {msg}", flush=True)


def run(cmd: list, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [str(c) for c in cmd]
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and p.returncode != 0:
        tail = (p.stderr or p.stdout or "")[-1500:]
        raise RuntimeError(f"command failed: {' '.join(cmd[:4])} ...\n{tail}")
    return p


def probe(path: Path) -> dict:
    """Width, height, fps and duration of a media file."""
    p = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path])
    data = json.loads(p.stdout)
    out = {"width": 0, "height": 0, "fps": 30.0, "duration": float(data.get("format", {}).get("duration", 0) or 0)}
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and not out["width"]:
            out["width"], out["height"] = int(s["width"]), int(s["height"])
            num, _, den = (s.get("avg_frame_rate") or "30/1").partition("/")
            try:
                out["fps"] = float(num) / float(den or 1) or 30.0
            except (ValueError, ZeroDivisionError):
                out["fps"] = 30.0
    return out


def slug(text: str, limit: int = 50) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text or "").strip("-").lower()
    return s[:limit].strip("-") or "clip"


def ts(sec: float) -> str:
    sec = max(0, int(sec))
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def even(x: float) -> int:
    return max(2, int(x) // 2 * 2)


def need(tool: str) -> None:
    import shutil

    if not shutil.which(tool):
        sys.exit(f"'{tool}' was not found on PATH. Install FFmpeg (it includes ffprobe) and try again.")
