"""Speech to text with word timings (faster-whisper, runs on CPU)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .util import log

FULL_CHUNK = 600  # seconds per piece when transcribing a whole stream


def pcm(audio: Path, start: float, dur: float) -> np.ndarray:
    """Decode a slice of any media file to 16 kHz mono floats, using FFmpeg."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}", "-i", str(audio),
           "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768


class Transcriber:
    def __init__(self, model: str = "small", language: str | None = None):
        self.model_name, self.language, self._model = model, language, None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            log(f"loading speech model '{self.model_name}' (first run downloads it)")
            self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        return self._model

    def window(self, audio: Path, start: float, end: float) -> dict:
        """Transcribe [start, end] of `audio`. Times in the result are absolute."""
        samples = pcm(audio, start, end - start)
        words, lines = [], []
        if len(samples) < 1600:
            return {"lines": lines, "words": words}
        segs, info = self._load().transcribe(samples, language=self.language, word_timestamps=True,
                                             vad_filter=True, beam_size=3)
        if self.language is None:
            self.language = info.language  # keep one language for the whole stream
        for s in segs:
            lines.append({"s": start + s.start, "e": start + s.end, "text": s.text.strip()})
            for w in s.words or []:
                if w.word.strip():
                    words.append({"w": w.word.strip(), "s": start + w.start, "e": start + w.end})
        return {"lines": lines, "words": words}

    def clip(self, media: Path, t0: float, dur: float) -> list[dict]:
        """Careful pass for the captions of one finished clip. Reads the clip's own
        high-quality sound, and drops words the model is not sure about: with game
        dialogue and the streamer talking at once, a wrong word is worse than none.
        Word times are relative to the start of the clip."""
        samples = pcm(media, t0, dur)
        if len(samples) < 1600:
            return []
        segs, _ = self._load().transcribe(samples, language=self.language, word_timestamps=True,
                                          vad_filter=True, beam_size=5, condition_on_previous_text=False)
        words = []
        for s in segs:
            if s.no_speech_prob > 0.6:
                continue
            for w in s.words or []:
                if w.word.strip() and w.probability >= 0.45:
                    words.append({"w": w.word.strip(), "s": w.start, "e": w.end})
        return words

    def full(self, audio: Path, duration: float) -> list[dict]:
        """Whole-stream transcript, in pieces so memory stays small (slow on CPU)."""
        out, t = [], 0.0
        while t < duration:
            samples = pcm(audio, t, FULL_CHUNK)
            if len(samples) >= 1600:
                segs, _ = self._load().transcribe(samples, language=self.language, vad_filter=True, beam_size=1)
                out += [{"s": t + s.start, "e": t + s.end, "text": s.text.strip()} for s in segs]
            t += FULL_CHUNK
        return out
