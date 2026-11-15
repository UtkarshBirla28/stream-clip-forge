"""Where the stream comes from: a link (Twitch, Kick, YouTube, ...) or a local file.

For a link we first fetch only the light parts (audio, chat replay, captions).
Video is downloaded later, and only for the moments that look good.
"""
from __future__ import annotations

import json
from pathlib import Path

from .util import log, probe

PAD = 3.0  # extra seconds downloaded on each side of a section


class Source:
    def __init__(self, target: str, want_chat: bool = True):
        self.target = target
        self.work: Path = Path(".")
        self._info: dict = {}
        self.want_chat = want_chat
        self.is_local = Path(target).exists()
        self.title = ""
        self.streamer = ""
        self.url = "" if self.is_local else target
        self.duration = 0.0
        self.audio: Path | None = None
        self.chat: list[tuple[float, str]] = []  # (seconds, message)
        self.captions: list[dict] = []  # [{s, e, text}] full-stream captions if the site has them

    # ---- light pass -----------------------------------------------------
    def identify(self) -> None:
        """Title, streamer and length. No downloads."""
        if self.is_local:
            p = Path(self.target)
            self.title, self.audio = p.stem, p
            self.duration = probe(p)["duration"]
            return
        import yt_dlp

        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(self.target, download=False)
        self._info = info
        self.title = info.get("title") or "stream"
        self.streamer = info.get("uploader") or info.get("channel") or ""
        self.duration = float(info.get("duration") or 0)
        self.url = info.get("webpage_url") or self.target

    def prepare(self, work: Path) -> None:
        self.work = work
        self.work.mkdir(parents=True, exist_ok=True)
        if self.is_local:
            return
        import yt_dlp

        info = self._info
        found = [p for p in self.work.glob("audio.*") if p.suffix != ".part"]
        if not found:
            log("downloading audio only (video comes later, just for the best moments)")
            opts = {
                "quiet": True, "no_warnings": True, "noprogress": True,
                "format": "worstaudio[abr>=40]/worstaudio/bestaudio/worst",
                "outtmpl": str(self.work / "audio.%(ext)s"),
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([self.target])
            found = [p for p in self.work.glob("audio.*") if p.suffix != ".part"]
        if not found:
            raise RuntimeError("could not download the audio for this link")
        self.audio = found[0]
        if not self.duration:
            self.duration = probe(self.audio)["duration"]
        self._extras(info)

    def _extras(self, info: dict) -> None:
        """Chat replay and site captions. Both are optional; failure is fine."""
        import yt_dlp

        langs = ["en.*", (info.get("language") or "en")]
        if self.want_chat:
            langs = ["live_chat", "rechat"] + langs
        if not list(self.work.glob("meta.*")):
            opts = {
                "quiet": True, "no_warnings": True, "noprogress": True, "ignoreerrors": True,
                "skip_download": True, "writesubtitles": True, "writeautomaticsub": True,
                "subtitleslangs": langs, "subtitlesformat": "json3/json/best",
                "outtmpl": str(self.work / "meta.%(ext)s"),
            }
            try:
                log("fetching chat replay and captions (if the site has them)")
                with yt_dlp.YoutubeDL(opts) as ydl:
                    ydl.download([self.target])
            except Exception as e:  # noqa: BLE001 - optional data
                log(f"no chat/captions available ({type(e).__name__})")
        for f in self.work.glob("meta.*"):
            name = f.name
            try:
                if ".live_chat." in name:
                    self.chat = _youtube_chat(f)
                elif ".rechat." in name:
                    self.chat = _twitch_chat(f)
                elif name.endswith(".json3") and not self.captions:
                    self.captions = _json3(f)
            except Exception as e:  # noqa: BLE001
                log(f"could not read {name}: {e}")
        if self.chat:
            log(f"chat replay: {len(self.chat)} messages")
        if self.captions:
            log(f"site captions: {len(self.captions)} lines")

    # ---- video for one moment ------------------------------------------
    def segment(self, start: float, end: float, name: str) -> tuple[Path, float]:
        """Return (video file, offset) so that `start` is at `offset` seconds in the file."""
        if self.is_local:
            return Path(self.target), start
        import yt_dlp
        from yt_dlp.utils import download_range_func

        seg_dir = self.work / "segs"
        seg_dir.mkdir(exist_ok=True)
        out = seg_dir / f"{name}.mp4"
        s0 = max(0.0, start - PAD)
        if not out.exists():
            opts = {
                "quiet": True, "no_warnings": True, "noprogress": True,
                # best picture up to 1440 on the short side (works for wide and
                # vertical streams); sharper source = sharper face cam close-up
                "format": "bv*+ba/b",
                "format_sort": ["res:1440", "fps"],
                "merge_output_format": "mp4",
                "download_ranges": download_range_func(None, [(s0, end + PAD)]),
                "force_keyframes_at_cuts": True,
                "outtmpl": str(seg_dir / f"{name}.%(ext)s"),
            }
            for attempt in (1, 2):  # section downloads fail now and then; one retry fixes most
                try:
                    with yt_dlp.YoutubeDL(opts) as ydl:
                        ydl.download([self.target])
                except Exception:  # noqa: BLE001
                    if attempt == 2:
                        raise
                if out.exists():
                    break
                for junk in seg_dir.glob(f"{name}.*"):
                    junk.unlink(missing_ok=True)
            if not out.exists():
                raise RuntimeError(f"section download failed for {name}")
        return out, start - s0


def _youtube_chat(path: Path) -> list[tuple[float, str]]:
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            act = json.loads(line)["replayChatItemAction"]
            t = int(act.get("videoOffsetTimeMsec", 0)) / 1000
            for a in act.get("actions", []):
                item = a.get("addChatItemAction", {}).get("item", {})
                r = item.get("liveChatTextMessageRenderer")
                if r:
                    text = "".join(x.get("text", "") for x in r.get("message", {}).get("runs", []))
                    out.append((t, text.strip()))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def _twitch_chat(path: Path) -> list[tuple[float, str]]:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(data, dict):
        data = data.get("comments", [])
    out = []
    for c in data:
        try:
            out.append((float(c["content_offset_seconds"]), str(c.get("message", {}).get("body", "")).strip()))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def _json3(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    out = []
    for ev in data.get("events", []):
        text = "".join(s.get("utf8", "") for s in ev.get("segs", []) or []).replace("\n", " ").strip()
        if text:
            s = ev.get("tStartMs", 0) / 1000
            out.append({"s": s, "e": s + ev.get("dDurationMs", 0) / 1000, "text": text})
    return out
