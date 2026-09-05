#!/usr/bin/env bash
#
# Bundles ../blackjack-trainer.jsx (React) plus React itself into a single
# self-contained ../index.html, the same shape as games/xiangqi/index.html:
# one static file, no CDN, no runtime transpile, no front matter.
#
#   npm install          # once, to get react + esbuild
#   npm run build:blackjack
#
set -euo pipefail
cd "$(dirname "$0")"
ESBUILD=../../../node_modules/.bin/esbuild

if [ ! -x "$ESBUILD" ]; then
  echo "esbuild not found — run 'npm install' at the repo root first." >&2
  exit 1
fi

BUNDLE=$(mktemp -t blackjack-bundle)
trap 'rm -f "$BUNDLE"' EXIT

"$ESBUILD" main.jsx \
  --bundle --minify --format=iife \
  --jsx=automatic \
  --define:process.env.NODE_ENV='"production"' \
  --outfile="$BUNDLE" --allow-overwrite

{
  cat head.html
  echo '<script>'
  cat "$BUNDLE"
  echo
  echo '</script>'
  echo '</body>'
  echo '</html>'
} > ../index.html

echo "built games/blackjack/index.html ($(wc -c < ../index.html | tr -d ' ') bytes)"
