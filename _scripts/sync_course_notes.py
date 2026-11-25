#!/usr/bin/env python3
"""Copy finished course notes into the site and regenerate _data/course_notes.yml.

Usage (from the repo root):
    python3 _scripts/sync_course_notes.py [path/to/transcribed]

The default source is ~/Documents/GoodNotes/transcribed. Every course folder
there with a notes.json (written by that folder's _guide/build_course.py) is
published:
  - its combined PDF goes to assets/pdf/course-notes/<slug>.pdf
  - each lecture's own PDF goes to assets/pdf/course-notes/<slug>/lectureNN.pdf,
    passed through Ghostscript, which shares the fonts the LaTeX output embeds
    several times over and so makes the file about a quarter of the size
  - its metadata and lecture index go into _data/course_notes.yml
"""
import datetime
import html
import json
import pathlib
import re
import shutil
import subprocess
import sys
import unicodedata

import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                   else "~/Documents/GoodNotes/transcribed").expanduser()
PDF_DIR = REPO / "assets/pdf/course-notes"
DATA = REPO / "_data/course_notes.yml"

SEASON = {"Spring": 1, "Summer": 2, "Fall": 3}


def term_key(term: str) -> tuple:
    season, year = term.split()
    return (int(year), SEASON[season])


def lecture_date(text: str) -> str:
    if not text:
        return ""
    return datetime.datetime.strptime(text, "%B %d, %Y").date().isoformat()


ACCENTS = {"'": "\u0301", '"': "\u0308", "`": "\u0300", "^": "\u0302",
           "~": "\u0303", "=": "\u0304", ".": "\u0307", "c": "\u0327", "v": "\u030c"}
TEXT_MACROS = {r"\S": "§", r"\&": "&amp;", r"\%": "%", "~": "&nbsp;",
               "``": "“", "''": "”", "---": "—", "--": "–"}
# LaTeX writes some macros to the .toc in expanded form that MathJax doesn't know.
MATH_MACROS = {r"\DOTSB \relbar \joinrel \rightarrow": r"\longrightarrow"}


def topic_html(title: str) -> str:
    """Turn a section title from the LaTeX .toc into HTML for the site.

    Accents and text macros become Unicode; $...$ math is kept for MathJax.
    """
    parts = re.split(r"(\$[^$]*\$)", title)
    out = []
    for i, part in enumerate(parts):
        if i % 2:  # math
            for tex, repl in MATH_MACROS.items():
                part = part.replace(tex, repl)
            out.append(html.escape(part, quote=False))
            continue
        part = html.escape(part, quote=False)
        part = re.sub(r"""\\([`'"^~=.cv])\s*\{?\\?([A-Za-z])\}?""",
                      lambda m: unicodedata.normalize("NFC", m.group(2) + ACCENTS[m.group(1)]),
                      part.replace("&quot;", '"'))
        for tex, repl in TEXT_MACROS.items():
            part = part.replace(tex, repl)
        out.append(part)
    return "".join(out)


# Section titles that say nothing about what a lecture covers: recaps, admin,
# and generic labels. They stay in the PDFs but are left out of the outline.
UNINFORMATIVE = re.compile(
    r"^(announcements|logistics|recall|summary|overview|outline|last time|introduction"
    r"|examples|remarks|lemmas|consequences|applications|where we are|two questions"
    r"|midterm problems|problem \d+|proof of question \d+|pfpl)$"
    r"|\b(recap|review|reminder|homework)\b|^updates to ",
    re.IGNORECASE)


def informative(title: str) -> bool:
    return not UNINFORMATIVE.search(title.strip())


def copy_lecture_pdf(src: pathlib.Path, dest: pathlib.Path) -> None:
    """Shrink a lecture PDF into the site, unless the copy there is current.

    Ghostscript output differs byte for byte on every run, so an up-to-date
    copy is left alone to keep git from seeing a change.
    """
    if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gs", "-q", "-sDEVICE=pdfwrite", "-dNOPAUSE", "-dBATCH",
                    "-o", str(dest), str(src)], check=True)


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    courses = []
    for notes_path in sorted(SRC.glob("*/notes.json")):
        notes = json.loads(notes_path.read_text())
        slug = notes["slug"]
        meta = yaml.safe_load((notes_path.parent / "course.yml").read_text())
        if meta.get("publish") is False:  # kept off the site by choice
            continue
        src_pdf = notes_path.parent / notes["pdf"]
        shutil.copyfile(src_pdf, PDF_DIR / f"{slug}.pdf")

        pdf_url = f"/assets/pdf/course-notes/{slug}.pdf"

        def lecture_url(lec: dict) -> str:
            # The lecture's own PDF when there is one, else its page in the combined PDF.
            src = notes_path.parent / f"{lec.get('file', '')}.pdf"
            if not lec.get("file") or not src.exists():
                return f"{pdf_url}#page={lec['page']}"
            copy_lecture_pdf(src, PDF_DIR / slug / src.name)
            return f"/assets/pdf/course-notes/{slug}/{src.name}"

        courses.append({
            "slug": slug,
            "number": notes["number"],
            "title": notes["title"],
            "term": notes["term"],
            "instructor": notes["instructor"],
            "course_url": notes.get("course_url"),
            "review_anchor": notes.get("review_anchor"),
            "pdf": pdf_url,
            "pages": notes["pages"],
            # Lectures in the notebook; more than len(lectures) while transcription is ongoing.
            "lectures_total": notes.get("lectures_total", len(notes["lectures"])),
            "lectures": [
                {
                    "number": lec["number"],
                    "date": lecture_date(lec["date"]),
                    "url": lecture_url(lec),
                    "topics": [topic_html(t) for t in lec["sections"] if informative(t)],
                }
                for lec in notes["lectures"]
            ],
        })
        print(f"{slug}: {len(notes['lectures'])} lectures, {notes['pages']} pages")

    courses.sort(key=lambda c: (term_key(c["term"]), c["number"]), reverse=True)
    header = ("# Generated by _scripts/sync_course_notes.py from the transcribed notes.\n"
              "# Don't edit by hand; re-run the script instead.\n")
    DATA.write_text(header + yaml.safe_dump(courses, sort_keys=False, allow_unicode=True,
                                            width=100))
    # Match the repo's Prettier style so the formatting check in CI passes.
    subprocess.run([str(REPO / "node_modules/.bin/prettier"), "--write", str(DATA)],
                   check=True, stdout=subprocess.DEVNULL)
    print(f"wrote {DATA.relative_to(REPO)} ({len(courses)} courses)")


if __name__ == "__main__":
    main()
