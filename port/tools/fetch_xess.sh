#!/usr/bin/env bash
# Downloads Intel's XeSS runtime (libxess.dll, Windows only) and its license into out/ (issue #62).
# The pinned XeSS SDK release zip is checked against its sha256; only bin/libxess.dll and
# LICENSE.txt (as XESS-LICENSE.txt) are kept. The Intel Simplified Software License allows
# shipping the unmodified DLL with the application. Without the DLL the game runs FSR 3.1.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
version=3.0.2
sha256=88b8a373f30e33f3558a77a93e634f11b8132fc3047ea1a8edeead32b8471990
url="https://github.com/intel/xess/releases/download/v$version/XeSS_SDK_$version.zip"
mkdir -p out
if [[ -s out/libxess.dll && -s out/XESS-LICENSE.txt ]]; then
    echo "XeSS: out/libxess.dll present"
    exit 0
fi
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
curl -fsSL --retry 3 -o "$work/xess.zip" "$url"
echo "$sha256  $work/xess.zip" | sha256sum -c --quiet -
unzip -q -j "$work/xess.zip" bin/libxess.dll LICENSE.txt -d "$work"
cp "$work/libxess.dll" out/libxess.dll
cp "$work/LICENSE.txt" out/XESS-LICENSE.txt
echo "XeSS: SDK $version libxess.dll in out/"
