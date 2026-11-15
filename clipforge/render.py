"""Turn one moment into a finished 1080x1920 Short with FFmpeg."""
from __future__ import annotations

import re
from pathlib import Path

from .util import even, probe, run

W, H = 1080, 1920
FACE_H = 608          # face panel height in the stacked layout
GAME_H = H - FACE_H   # gameplay panel height


def _t(sec: float) -> str:
    sec = max(0.0, sec)
    return f"{int(sec // 3600)}:{int(sec % 3600 // 60):02d}:{sec % 60:05.2f}"


def _clean(text: str) -> str:
    return text.replace("{", "(").replace("}", ")").replace("\n", " ").strip()


def write_ass(path: Path, words: list[dict], hook: str, dur: float, stacked: bool) -> None:
    """Word-by-word captions (the spoken word turns yellow) plus the hook banner."""
    cap_y = 1500 if stacked else 1560
    hook_y = FACE_H if stacked else 250
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Arial,82,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,7,2,5,70,70,0,1
Style: Hook,Arial,58,&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,0,3,16,0,5,90,90,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    if hook:
        ev.append(f"Dialogue: 1,{_t(0)},{_t(dur)},Hook,,0,0,0,,{{\\pos({W // 2},{hook_y})}}{_clean(hook)}")
    groups, cur = [], []
    for w in words:
        gap = cur and w["s"] - cur[-1]["e"] > 0.7
        if cur and (len(cur) >= 3 or gap or sum(len(x["w"]) for x in cur) + len(w["w"]) > 18):
            groups.append(cur)
            cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    for gi, g in enumerate(groups):
        g_end = min(g[-1]["e"] + 0.25, groups[gi + 1][0]["s"] if gi + 1 < len(groups) else dur)
        for i, w in enumerate(g):
            a = w["s"]
            b = g[i + 1]["s"] if i + 1 < len(g) else g_end
            if b <= a:
                continue
            text = " ".join(
                (f"{{\\c&H00F6FF&}}{_clean(x['w'])}{{\\c&HFFFFFF&}}" if j == i else _clean(x["w"]))
                for j, x in enumerate(g))
            ev.append(f"Dialogue: 0,{_t(a)},{_t(b)},Cap,,0,0,0,,{{\\pos({W // 2},{cap_y})}}{text}")
    path.write_text(head + "\n".join(ev) + "\n", encoding="utf-8")


def audio_level(path: Path) -> float | None:
    """Average loudness of a file in dB, or None if it has no sound track.
    The stream's sound is one mixed track (streamer voice + game), kept as is."""
    p = run(["ffmpeg", "-v", "info", "-i", path, "-vn", "-af", "volumedetect", "-f", "null", "-"], check=False)
    m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", p.stderr or "")
    return float(m.group(1)) if m else None


def render(video: Path, offset: float, dur: float, out_dir: Path, words: list[dict], hook: str,
           face: tuple[int, int, int, int] | None, gameplay: str = "crop") -> Path:
    """`words` use times relative to the start of the clip."""
    info = probe(video)
    sw, sh = info["width"], info["height"]
    fps = min(60, max(24, round(info["fps"])))
    vertical = sh >= 1.4 * sw  # the stream is already phone-shaped: keep its own layout
    stacked = face is not None and not vertical
    write_ass(out_dir / "subs.ass", words, hook, dur, stacked)

    if vertical:
        graph = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos,"
                 f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,ass=subs.ass,fps={fps}[v]")
    elif stacked:
        fx, fy, fw, fh = face
        if gameplay == "fit":  # whole frame, smaller, on a blurred fill
            game = (f"[b]split=2[b1][b2];[b1]scale={W}:{GAME_H}:force_original_aspect_ratio=increase,"
                    f"crop={W}:{GAME_H},boxblur=20:4,eq=brightness=-0.12[bg];"
                    f"[b2]scale={W}:-2[fg];[bg][fg]overlay=0:(H-h)/2[game]")
        else:  # centre of the frame, filling the panel
            gw = min(sw, even(sh * W / GAME_H))
            game = f"[b]crop={gw}:{sh}:{even((sw - gw) / 2)}:0,scale={W}:{GAME_H}:flags=lanczos[game]"
        # the face cam is a small box blown up to full width, so sharpen it a little
        sharp = ",unsharp=5:5:0.8:5:5:0.0" if W / fw > 1.5 else ""
        graph = (f"[0:v]split=2[a][b];[a]crop={fw}:{fh}:{fx}:{fy},scale={W}:{FACE_H}:flags=lanczos{sharp}[face];"
                 f"{game};[face][game]vstack,setsar=1,ass=subs.ass,fps={fps}[v]")
    else:
        cw = min(sw, even(sh * 4 / 3))  # centre 4:3 slice, so the picture is bigger on a phone
        graph = (f"[0:v]split=2[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
                 f"boxblur=24:4,eq=brightness=-0.15[bg];[b]crop={cw}:{sh}:{even((sw - cw) / 2)}:0,"
                 f"scale={W}:-2:flags=lanczos[fg];[bg][fg]overlay=0:(H-h)/2,setsar=1,ass=subs.ass,fps={fps}[v]")

    out = out_dir / "short.mp4"
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{offset:.3f}", "-t", f"{dur:.3f}", "-i", video,
         "-filter_complex", graph, "-map", "[v]", "-map", "0:a?",
         "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart", "short.mp4"],
        cwd=out_dir)
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{min(dur * 0.5, max(0, dur - 1)):.2f}", "-i", "short.mp4",
         "-frames:v", "1", "-q:v", "3", "thumb.jpg"], cwd=out_dir)
    (out_dir / "subs.ass").unlink(missing_ok=True)
    return out
