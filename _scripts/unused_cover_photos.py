#!/usr/bin/env python3
"""List photos in assets/img/posts that no post, draft, summary or page uses yet.

Run from the repo root:  python3 _scripts/unused_cover_photos.py
Rewrites _scripts/unused_cover_photos.md.
"""
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
PHOTOS = ROOT / "assets/img/posts"
OUT = ROOT / "_scripts/unused_cover_photos.md"
SEARCH = ["_posts", "_drafts", "_summaries", "_pages", "_projects", "_news", "_data", "_layouts", "_includes"]
TEXT_EXT = {".md", ".markdown", ".html", ".yml", ".yaml", ".liquid"}

text = ""
for d in SEARCH:
    for f in (ROOT / d).rglob("*"):
        if f.is_file() and f.suffix in TEXT_EXT:
            text += f.read_text(errors="ignore") + "\n"

# date each photo was first committed
added = {}
log = subprocess.run(
    ["git", "log", "--diff-filter=A", "--name-only", "--format=@%ad", "--date=short", "--", "assets/img/posts"],
    cwd=ROOT, capture_output=True, text=True).stdout
date = None
for line in log.splitlines():
    if line.startswith("@"):
        date = line[1:]
    elif line.strip():
        added[pathlib.Path(line).name] = date

unused = []
for f in sorted(PHOTOS.glob("*.webp")):
    if not re.search(r"(?<![\w-])" + re.escape(f.stem) + r"\.(webp|avif|jpe?g|png|heic)", text, re.I):
        unused.append((added.get(f.name, "uncommitted"), f.name))

by_date = {}
for d, name in unused:
    by_date.setdefault(d, []).append(name)

lines = ["# Unused cover photos", "",
         f"{len(unused)} photos in `assets/img/posts/` that nothing on the site references yet.",
         "Regenerate with `python3 _scripts/unused_cover_photos.py`.", ""]
for d in sorted(by_date, reverse=True):
    lines += [f"## Added {d}", ""] + [f"- {n}" for n in by_date[d]] + [""]
OUT.write_text("\n".join(lines))
print(f"{len(unused)} unused, written to {OUT.relative_to(ROOT)}")
