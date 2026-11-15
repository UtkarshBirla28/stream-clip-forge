"""Fast checks plus one end-to-end run on a video made by FFmpeg (no network, no LLM)."""
import json
import subprocess

import numpy as np

from clipforge import cli, judge, render, signals


def test_candidates_find_the_spikes():
    db = np.full(600, -40.0) + np.random.default_rng(0).normal(0, 1, 600)
    db[200:204] += 25
    db[450:453] += 20
    total, az, cz = signals.score(db, None)
    got = sorted(c["t"] for c in signals.candidates(total, az, cz, 2))
    assert abs(got[0] - 201) <= 3 and abs(got[1] - 451) <= 3


def test_chat_spike_counts():
    db = np.full(600, -40.0)
    chat = signals.chat_rate([(300 + i / 10, "W") for i in range(200)], 600)
    total, az, cz = signals.score(db, chat)
    assert abs(signals.candidates(total, az, cz, 1)[0]["t"] - 305) <= 15


def test_fit_keeps_clip_inside_window_and_limits():
    s, e = judge._fit(-5, 500, {"w0": 100.0, "w1": 160.0}, 15, 50)
    assert s >= 100 and e <= 160 and 15 <= e - s <= 50


def test_ass_has_hook_and_words(tmp_path):
    words = [{"w": "no", "s": 0.5, "e": 0.8}, {"w": "way", "s": 0.8, "e": 1.2}]
    render.write_ass(tmp_path / "s.ass", words, "He lost {everything}", 5, True)
    text = (tmp_path / "s.ass").read_text(encoding="utf-8")
    assert "He lost (everything)" in text and text.count("Dialogue:") == 3


def test_end_to_end_on_synthetic_video(tmp_path):
    video = tmp_path / "stream.mp4"
    loud = "between(t,60,64)+between(t,130,133)"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=180",
                    "-f", "lavfi", "-i", f"aevalsrc='(0.01+0.6*({loud}))*sin(2*PI*440*t)':d=180:s=44100",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(video)],
                   check=True)
    code = cli.main([str(video), "-n", "2", "-o", str(tmp_path / "out"), "--llm", "none", "--no-transcribe"])
    assert code == 0
    clips = json.loads((tmp_path / "out" / "stream" / "clips.json").read_text(encoding="utf-8"))
    assert len(clips) == 2
    for c in clips:
        d = tmp_path / "out" / "stream" / c["dir"]
        p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", str(d / "short.mp4")],
                           capture_output=True, text=True, check=True)
        assert p.stdout.strip() == "1080,1920"
        assert (d / "post.txt").exists() and (d / "thumb.jpg").exists()
    assert (tmp_path / "out" / "stream" / "index.html").exists()
