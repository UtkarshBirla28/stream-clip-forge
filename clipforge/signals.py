"""Find moments where something happened, without watching the video.

Two signals: how loud the stream suddenly gets (screams, laughing, shouting)
and how fast chat starts moving. Both are compared with what is normal for
that part of the stream, so a quiet streamer and a loud one are treated alike.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

SR = 8000
CHAT_LAG = 5  # chat reacts a few seconds after the moment


def loudness(audio: Path) -> np.ndarray:
    """Loudness in dB for every second of the file."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(audio), "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    out, chunk = [], SR * 2 * 60  # one minute of 16-bit samples
    buf = b""
    while True:
        data = proc.stdout.read(chunk)
        if not data:
            break
        buf += data
        n = len(buf) // (SR * 2)
        if n:
            x = np.frombuffer(buf[: n * SR * 2], dtype=np.int16).astype(np.float32).reshape(n, SR) / 32768
            out.append(np.sqrt((x**2).mean(axis=1)))
            buf = buf[n * SR * 2:]
    proc.wait()
    rms = np.concatenate(out) if out else np.zeros(1, dtype=np.float32)
    return np.clip(20 * np.log10(rms + 1e-7), -70, 0)


def chat_rate(chat: list[tuple[float, str]], seconds: int) -> np.ndarray:
    rate = np.zeros(seconds, dtype=np.float32)
    for t, _ in chat:
        if 0 <= t < seconds:
            rate[int(t)] += 1
    return rate


def local_z(x: np.ndarray, block: int = 120) -> np.ndarray:
    """How unusual each second is compared with the minutes around it."""
    n = len(x)
    if n < 4:
        return np.zeros(n)
    nb = max(1, int(np.ceil(n / block)))
    med = np.array([np.median(x[i * block:(i + 1) * block]) for i in range(nb)])
    centres = np.array([min(n - 1, i * block + block / 2) for i in range(nb)])
    resid = x - np.interp(np.arange(n), centres, med)
    mad = np.median(np.abs(resid - np.median(resid))) * 1.4826 + 1e-3
    return resid / mad


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(x, np.ones(w) / w, mode="same") if len(x) >= w else x


def score(db: np.ndarray, chat: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    az = smooth(local_z(db), 3)
    cz = np.zeros_like(az)
    if chat is not None and chat.sum() > 20:
        z = local_z(smooth(chat, 8))
        cz[: len(z) - CHAT_LAG] = z[CHAT_LAG:]
    total = np.clip(az, 0, None) + 1.5 * np.clip(cz, 0, None)
    return total, az, cz


def candidates(total: np.ndarray, az: np.ndarray, cz: np.ndarray, k: int,
               sep: int = 45, extra: list[dict] | None = None) -> list[dict]:
    """Top `k` moments, at least `sep` seconds apart. `extra` are moments an
    LLM spotted in the transcript; they get a boost so they make the shortlist."""
    n = len(total)
    s = total.copy()
    s[:5] = 0
    s[max(0, n - 5):] = 0
    pool = []
    order = np.argsort(s)[::-1]
    for i in order[: max(k * 60, 600)]:
        pool.append({"t": float(i), "score": float(s[i]), "why": ""})
    top = float(s.max()) if n else 1.0
    for e in extra or []:
        t = int(min(max(e["t"], 0), n - 1))
        pool.append({"t": float(t), "score": float(s[max(0, t - 5): t + 6].max()) + 0.6 * top, "why": e.get("why", "")})
    pool.sort(key=lambda c: c["score"], reverse=True)
    picked: list[dict] = []
    for c in pool:
        if c["score"] <= 0:
            break
        if all(abs(c["t"] - p["t"]) >= sep for p in picked):
            i = int(c["t"])
            c["audio_z"], c["chat_z"] = round(float(az[i]), 1), round(float(cz[i]), 1)
            c["score"] = round(c["score"], 2)
            picked.append(c)
            if len(picked) >= k:
                break
    return picked
