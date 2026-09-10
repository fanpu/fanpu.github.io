#!/usr/bin/env python3
"""Convert source images under the current directory to .webp.

Line art (plots, diagrams, screenshots) is encoded lossless at its native
resolution, so small text like axis labels and heatmap cell values stays crisp.
Photographs are encoded lossy and capped at 1024px wide, which is the
long-standing convention for this site.

The old version of this script ran `convert <src> -resize 1024\\> <dst>` on
everything with no quality flag. That silently downscaled large plots (a
2773px-wide figure became 1024px) and encoded them lossy at ImageMagick's
default quality, which destroys small text. It went unnoticed for a long time
because until recently every figure here was already narrower than 1024px.

Usage:
    python ../convert_avif.py            # skip images that already have a .webp
    python ../convert_avif.py --force    # re-encode everything

Only files with a source image are written, so hand-made .webp files with no
.png/.jpg counterpart are left alone.
"""

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile

TARGET_SUFFIX = ".webp"

# Sources that are already lossy are always treated as photos: re-encoding a
# JPEG losslessly just preserves its compression artifacts at a much larger
# size. This also keeps low-colour photographs (a dark night sky, a snowy
# mountain) out of the line-art branch, which the colour count alone gets wrong.
PHOTO_EXTS = {".jpg", ".JPG", ".jpeg", ".JPEG", ".tiff", ".TIFF"}
LINE_ART_EXTS = {".png", ".PNG"}

# Plots and diagrams here run from ~60 to ~46,000 unique colours; photographs
# saved as PNG start around 80,000. 50,000 sits in the gap.
LINE_ART_MAX_COLORS = 50_000

# If a "line art" file balloons past this when encoded losslessly it is really a
# photograph that slipped under the colour threshold, so fall back to the photo
# branch rather than committing a huge file.
LOSSLESS_SIZE_GUARD = 1_500_000

PHOTO_MAX_WIDTH = 1024
PHOTO_QUALITY = 82
LINE_ART_LOSSY_QUALITY = 95


def run(cmd):
    """Run a command, returning True on success."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ! {cmd[0]} failed: {result.stderr.strip().splitlines()[-1:]}")
        return False
    return True


def identify(src, fmt):
    """Read a single ImageMagick format field from an image."""
    result = subprocess.run(
        ["identify", "-format", fmt, str(src)], capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    # Multi-frame images emit one line per frame; the first is enough.
    return result.stdout.splitlines()[0] if result.stdout else None


def is_line_art(src):
    if src.suffix in PHOTO_EXTS:
        return False
    if src.suffix not in LINE_ART_EXTS:
        return False
    colors = identify(src, "%k")
    if colors is None:
        return False
    return int(colors) < LINE_ART_MAX_COLORS


def encode_line_art(src, dst):
    """Encode at native resolution, keeping the smaller of lossless and q95.

    Both candidates are full resolution, so choosing between them only trades
    bytes and never sharpness. Note this comparison is deliberately confined to
    this branch: a downscaled lossy encode is often smaller still, and picking
    it would reintroduce exactly the problem this script exists to avoid.
    """
    with tempfile.TemporaryDirectory() as tmp:
        lossless = pathlib.Path(tmp) / "lossless.webp"
        lossy = pathlib.Path(tmp) / "lossy.webp"

        if not run(["cwebp", "-quiet", "-lossless", "-z", "9", str(src), "-o", str(lossless)]):
            return False

        if lossless.stat().st_size > LOSSLESS_SIZE_GUARD:
            print(f"  lossless would be {lossless.stat().st_size / 1e6:.1f}MB; treating as photo")
            return encode_photo(src, dst)

        ok_lossy = run(
            ["cwebp", "-quiet", "-q", str(LINE_ART_LOSSY_QUALITY), str(src), "-o", str(lossy)]
        )
        best = lossless
        if ok_lossy and lossy.stat().st_size < lossless.stat().st_size:
            best = lossy

        shutil.copyfile(best, dst)
        print(f"  line art, native res, {best.stem} -> {dst.stat().st_size / 1024:.0f}KB")
        return True


def encode_photo(src, dst):
    """Encode lossy, shrinking to PHOTO_MAX_WIDTH but never enlarging."""
    cmd = ["cwebp", "-quiet", "-q", str(PHOTO_QUALITY)]

    width = identify(src, "%w")
    if width is not None and int(width) > PHOTO_MAX_WIDTH:
        # 0 for height keeps the aspect ratio.
        cmd += ["-resize", str(PHOTO_MAX_WIDTH), "0"]

    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "photo.webp"
        if not run(cmd + [str(src), "-o", str(out)]):
            return False
        shutil.copyfile(out, dst)
        print(f"  photo, q{PHOTO_QUALITY} -> {dst.stat().st_size / 1024:.0f}KB")
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-encode even if the .webp already exists"
    )
    args = parser.parse_args()

    for tool in ("cwebp", "identify"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} not found on PATH (brew install webp imagemagick)")

    converted = skipped = failed = 0
    exts = PHOTO_EXTS | LINE_ART_EXTS

    for src in sorted(pathlib.Path(".").rglob("*")):
        if not src.is_file() or src.suffix not in exts:
            continue

        dst = src.with_suffix(TARGET_SUFFIX)
        if dst.exists() and not args.force:
            skipped += 1
            continue

        print(f"Converting {src}")
        ok = encode_line_art(src, dst) if is_line_art(src) else encode_photo(src, dst)
        if ok:
            converted += 1
        else:
            failed += 1

    print(f"\n{converted} converted, {skipped} skipped, {failed} failed")


# To convert from gif to webp:
# ffmpeg -i input.gif -vcodec webp -loop 0 -pix_fmt yuva420p output.webp

if __name__ == "__main__":
    main()
