#!/usr/bin/env bash
# Download the T1 archive and demographic workbook; no data belongs in a release.
set -euo pipefail
if (($# != 1)); then printf 'usage: bash scripts/download-ixi.sh DIRECTORY\n' >&2; exit 2; fi
umask 077
out=$1
mkdir -p "$out"
url=https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI
for name in IXI-T1.tar IXI.xls; do
    [[ ! -e "$out/$name" && ! -e "$out/$name.partial" ]]
    curl --fail --location --retry 2 --output "$out/$name.partial" "$url/$name"
    [[ -s "$out/$name.partial" ]]
    mv "$out/$name.partial" "$out/$name"
done
printf '%s\n' 'IXI Dataset, Brain Development: https://brain-development.org/ixi-dataset/' \
    'Creative Commons CC BY-SA 3.0: https://creativecommons.org/licenses/by-sa/3.0/' \
    > "$out/SOURCE.txt"
