"""One HTML page to watch every clip and copy its title and description."""
from __future__ import annotations

import html
from pathlib import Path

PAGE = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Clips - {title}</title>
<style>
body{{margin:0;font:15px/1.5 system-ui,sans-serif;background:#111;color:#eee}}
header{{padding:20px 24px;border-bottom:1px solid #2a2a2a}} h1{{margin:0;font-size:20px}} header p{{margin:4px 0 0;color:#999}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:20px;padding:24px}}
article{{background:#1b1b1b;border:1px solid #2a2a2a;border-radius:12px;overflow:hidden}}
video{{width:100%;aspect-ratio:9/16;background:#000;display:block}}
.b{{padding:14px}} h2{{margin:0 0 6px;font-size:16px}} .m{{color:#999;font-size:13px;margin:0 0 10px}}
pre{{white-space:pre-wrap;background:#111;border:1px solid #2a2a2a;border-radius:8px;padding:10px;font:13px/1.45 system-ui;margin:0 0 10px}}
button{{background:#f5d90a;color:#111;border:0;border-radius:8px;padding:8px 12px;font-weight:600;cursor:pointer;margin-right:6px}}
</style>
<header><h1>{title}</h1><p>{n} clips, best first. Watch each one before you post it.</p></header>
<main>{cards}</main>
<script>
document.querySelectorAll('button[data-copy]').forEach(b=>b.onclick=()=>{{
  navigator.clipboard.writeText(document.getElementById(b.dataset.copy).textContent);
  const t=b.textContent;b.textContent='Copied';setTimeout(()=>b.textContent=t,1200)}});
</script>"""

CARD = """<article><video src="{dir}/short.mp4" poster="{dir}/thumb.jpg" controls preload="none"></video><div class="b">
<h2 id="t{i}">{title}</h2><p class="m">Score {score} &middot; {length}s &middot; at {at} in the stream<br>{reason}</p>
<pre id="d{i}">{desc}</pre>
<button data-copy="t{i}">Copy title</button><button data-copy="d{i}">Copy description</button></div></article>"""


def write(out: Path, title: str, clips: list[dict]) -> Path:
    e = html.escape
    cards = "".join(CARD.format(i=i, dir=e(c["dir"]), title=e(c["title"]), score=round(c["rank_score"]),
                                length=round(c["end"] - c["start"]), at=e(c["at"]), reason=e(c["reason"]),
                                desc=e(c["post_description"])) for i, c in enumerate(clips))
    page = out / "index.html"
    page.write_text(PAGE.format(title=e(title), n=len(clips), cards=cards), encoding="utf-8")
    return page
