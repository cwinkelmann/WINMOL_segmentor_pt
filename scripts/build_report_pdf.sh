#!/usr/bin/env bash
# Assemble the read-through report as a single PDF.
#
# Sections are concatenated in reading order, not merged, so each source document stays
# the single place its numbers live — the same rule the HTML report follows. Run from the
# repo root; pandoc resolves image paths relative to --resource-path.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${1:-docs/WINMOL-report.pdf}
PARTS=(
  docs/report-front.md
  docs/data-inventory.md
  docs/reder-method-gaps-closed.md
  docs/scale-augmentation-results.md
  docs/scale-augmentation-loso.md
  docs/process.md
)
for f in "${PARTS[@]}"; do [ -f "$f" ] || { echo "missing: $f" >&2; exit 1; }; done

# Two substitutions, both because the output would be silently wrong otherwise:
#   - `[[wiki links]]` are for the memory index and render as literal brackets.
#   - Helvetica Neue has no U+2192/U+2194, so arrows vanish rather than failing the build.
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
# Cap figure size: several are 2000 px tall and overflow the text block otherwise.
cat > "$TMP/header.tex" <<'TEX'
\usepackage{graphicx}
\setkeys{Gin}{width=\linewidth,height=0.80\textheight,keepaspectratio}
TEX

SRC=()
for f in "${PARTS[@]}"; do
  b="$TMP/$(basename "$f")"
  sed -E -e 's/\[\[([^]]+)\]\]/\1/g' \
         -e 's/\xe2\x86\x94/<->/g' -e 's/\xe2\x86\x92/->/g' "$f" > "$b"
  SRC+=("$b")
done

pandoc "${SRC[@]}" \
  --pdf-engine=xelatex \
  --resource-path=.:docs \
  --highlight-style=tango \
  -V colorlinks=true \
  -V mainfont="Helvetica Neue" \
  -V monofont="Menlo" \
  -H "$TMP/header.tex" \
  -o "$OUT"

echo "$OUT  ($(du -h "$OUT" | cut -f1))"
