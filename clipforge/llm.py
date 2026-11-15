"""Ask Claude a question and get text back.

Backends:
  claude-cli  the `claude` command (Claude Code). Uses your Claude login, no API key.
  api         the Anthropic API through the official SDK. Needs credentials.
  none        no LLM; clipforge falls back to simple rules.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
from pathlib import Path

DEFAULT_MODEL = "claude-opus-5-5"


def pick_backend(choice: str) -> str:
    if choice != "auto":
        return choice
    if shutil.which("claude"):
        return "claude-cli"
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "api"
    return "none"


class LLM:
    def __init__(self, backend: str = "auto", model: str | None = None, cwd: Path | None = None):
        self.backend = pick_backend(backend)
        self.model = model
        self.cwd = cwd

    @property
    def enabled(self) -> bool:
        return self.backend != "none"

    def ask(self, prompt: str, images: list[Path] | None = None) -> str:
        if self.backend == "claude-cli":
            return self._cli(prompt, images or [])
        if self.backend == "api":
            return self._api(prompt, images or [])
        raise RuntimeError("no LLM backend is enabled")

    def ask_json(self, prompt: str, images: list[Path] | None = None) -> dict:
        text = self.ask(prompt, images)
        a, b = text.find("{"), text.rfind("}")
        if a < 0 or b <= a:
            raise ValueError("the model did not return JSON")
        return json.loads(text[a: b + 1])

    # ---- Claude Code CLI ------------------------------------------------
    def _cli(self, prompt: str, images: list[Path]) -> str:
        exe = shutil.which("claude")
        if not exe:
            raise RuntimeError("the 'claude' command was not found")
        cmd = [exe, "-p", "--output-format", "text"]
        if self.model:
            cmd += ["--model", self.model]
        if images:
            cmd += ["--allowedTools", "Read"]
            listing = "\n".join(f"- {p}" for p in images)
            prompt += ("\n\nBefore you answer, open each of these image files with the Read tool "
                       f"(file names match the moment ids):\n{listing}\n")
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", cwd=self.cwd, timeout=1800)
        if p.returncode != 0:
            raise RuntimeError(f"claude CLI failed: {(p.stderr or p.stdout)[-600:]}")
        return p.stdout

    # ---- Anthropic API --------------------------------------------------
    def _api(self, prompt: str, images: list[Path]) -> str:
        import anthropic

        content: list[dict] = []
        for img in images:
            content.append({"type": "text", "text": f"Image for {img.stem}:"})
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": base64.standard_b64encode(img.read_bytes()).decode("ascii")}})
        content.append({"type": "text", "text": prompt})
        client = anthropic.Anthropic()
        try:
            # `fallbacks` lets the API retry on another model if a safety
            # classifier declines the request, instead of just stopping.
            resp = client.beta.messages.create(
                model=self.model or DEFAULT_MODEL,
                max_tokens=16000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{"role": "user", "content": content}],
            )
        except anthropic.AuthenticationError as e:
            raise RuntimeError("Anthropic credentials were rejected") from e
        except anthropic.RateLimitError as e:
            raise RuntimeError("Anthropic rate limit hit; try again in a minute") from e
        except anthropic.APIStatusError as e:
            raise RuntimeError(f"Anthropic API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise RuntimeError("could not reach the Anthropic API") from e
        if resp.stop_reason == "refusal":
            raise RuntimeError("the model declined this request")
        return "".join(b.text for b in resp.content if b.type == "text")
