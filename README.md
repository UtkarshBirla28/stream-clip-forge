# stream-clip-forge

Give it a stream link. Get back ready-to-post Shorts.

`clipforge` finds the best moments in a long game stream and turns each one into a
1080x1920 vertical video with the streamer's face cam on top, the gameplay below,
word-by-word captions, a hook line, a title, a description and hashtags. It works for
any game and any site that [yt-dlp](https://github.com/yt-dlp/yt-dlp) can read
(Twitch, Kick, YouTube and many more), or a video file on your disk.

```
clipforge "https://www.twitch.tv/videos/123456789" --game "GTA 6" -n 6
```

## What it does

1. **Light download.** Fetches only the audio, the chat replay and the site's captions.
   The full video is never downloaded.
2. **Finds moments.** Looks for seconds where the stream suddenly gets loud (screams,
   laughing) and where chat suddenly speeds up. If the site has captions, Claude also
   reads the whole transcript and points at moments the other two signals would miss.
3. **Fetches video for the shortlist only**, about one minute per moment.
4. **Transcribes** each moment with word timings (faster-whisper, on your CPU).
5. **Claude judges.** It gets the speech, the chat and four frames from each moment,
   throws out false alarms, keeps the best, sets the exact start and end, and writes
   the hook, title, description and hashtags.
6. **Finds the face cam.** A face that stays in the same spot across the stream is the
   webcam; faces inside the game move around and are ignored.
7. **Renders the Short** with FFmpeg: face on top, gameplay below, captions burned in,
   audio levelled to -14 LUFS. With no face cam it uses the whole picture on a blurred fill.
   A stream that is already vertical keeps its own layout.
8. **Writes a review page** (`index.html`) where you watch every clip and copy its text.

## Install

You need Python 3.10+ and [FFmpeg](https://ffmpeg.org/download.html) on your PATH.

```
git clone https://github.com/UtkarshBirla28/stream-clip-forge
cd stream-clip-forge
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -e .
```

### The judge

`clipforge` picks the first of these that is available:

| Backend | Needs | Notes |
|---|---|---|
| `claude-cli` | [Claude Code](https://claude.com/claude-code) installed and logged in | Uses your Claude login. No API key. |
| `api` | `pip install -e ".[api]"` and `ANTHROPIC_API_KEY` | Billed per use. Default model `claude-opus-5-5`; change with `--model`. |
| `none` | nothing | Simple rules: strongest spikes win, plain titles. Good for a quick test. |

Force one with `--llm claude-cli`, `--llm api` or `--llm none`.

## Use

```
clipforge <link or file> [options]
```

| Option | What it does |
|---|---|
| `-n 6` | How many clips to make |
| `--game "GTA 6"` | Game name, used in titles and tags |
| `--streamer "Name"` | Streamer name (taken from the link if you leave it out) |
| `--min 15 --max 50` | Clip length limits in seconds |
| `--layout auto\|stack\|full` | `stack` = face on top, gameplay below. `full` = whole picture |
| `--facecam x,y,w,h` | Face cam box in pixels, if auto-detect misses it |
| `--gameplay crop\|fit` | Fill the panel with the centre of the game, or show the whole frame |
| `--whisper small` | Speech model: `tiny`, `base`, `small`, `medium`. Bigger is better and slower |
| `--caption-model turbo` | Stronger speech model used only for the captions of the final clips. `none` reuses the first pass |
| `--keep-work` | Keep the downloaded audio and video sections. By default they are deleted when the run ends |
| `--language hi` | Spoken language, if auto-detect gets it wrong |
| `--full-transcript` | Transcribe the whole stream for the transcript scan (slow on CPU) |
| `--no-vision` | Do not send frames to Claude (faster, cheaper, less accurate) |
| `--no-chat` | Skip the chat replay |
| `--credit` | Add "Clip from ... stream: link" to each description |
| `-o out` | Output folder |

## Output

```
out/<stream-title>/
  index.html                 review page: watch, then copy title and description
  clips.json                 everything about every clip
  01_<clip-title>/
    short.mp4                1080x1920, H.264 + AAC, ready to upload
    thumb.jpg
    post.txt                 title and description to paste
    meta.json                score, reason, times, hook, hashtags
```

Everything the tool downloaded (audio, video sections) is deleted when the run ends, so
only the finished clips stay on disk. Pass `--keep-work` to keep it in `_work/`; a
second run then reuses it instead of downloading again.

Captions come from the clip's sound, which is one mixed track: the streamer and the
game together. When both talk at once the speech model can only follow one of them,
so the tool drops words it is unsure about instead of showing wrong ones.

## Limits

- Chat replay works for YouTube and Twitch. For other sites the tool uses loudness and
  the transcript only.
- The transcript scan runs when the site has captions (YouTube usually does) or when
  you pass `--full-transcript`.
- The face cam is looked for in every clip, so it may sit in a different place in each
  part of a stream. If the streamer changes scene in the middle of a clip (cam goes
  full screen, say), the crop can be wrong for that part: check the clip, and use
  `--layout full` or `--facecam x,y,w,h` if needed.
- The face cam is a small box enlarged to full width, so it is only as sharp as the
  stream. 1080p or better sources look best.
- The caption model (`turbo`) needs about 3 GB of free memory. On a small machine use
  `--caption-model small` or `--caption-model none`.
- The tool does not upload anything. You post the clips yourself.
- Game radio or other music in a clip can still trigger a music claim on upload.

## Rights

Only clip streams you have permission to clip. A streamer's video is their work, and
the sites you post to have their own rules about reused content. This tool makes the
clips; what you are allowed to post is on you.

## Tests

```
pip install -e ".[dev]"
pytest
```

The tests make a small video with FFmpeg and run the whole pipeline on it, so they
need no network and no Claude login.

## License

MIT
