#!/usr/bin/env bash
# Rebuild the social preview from the voxel-brain icon used in the banner.
set -euo pipefail
out=$(cd "$(dirname "$0")" && pwd)
magick -size 1280x640 xc:white \( "$out/icon.png" -trim +repage -resize 464x464 \) \
    -gravity center -composite -strip "PNG24:$out/social-preview.png"
