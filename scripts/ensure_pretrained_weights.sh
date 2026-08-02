#!/usr/bin/env bash
# Ensure pretrained/*/pytorch_model.bin exist as real files.
# Copies from a local release bundle if weights are missing.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUNDLE="${PRETRAINED_BUNDLE:-${REPO_ROOT}/pretrained_release}"

declare -A SRC=(
  [ted-hls-12b768]="${BUNDLE}/ted-hls-12b768/pytorch_model.bin"
  [msm-hls-12b768]="${BUNDLE}/msm-hls-12b768/pytorch_model.bin"
  [ntp-hls-12b768]="${BUNDLE}/ntp-hls-12b768/pytorch_model.bin"
)

for name in ted-hls-12b768 msm-hls-12b768 ntp-hls-12b768; do
  dest_dir="${REPO_ROOT}/pretrained/${name}"
  dest="${dest_dir}/pytorch_model.bin"
  mkdir -p "$dest_dir"
  if [ -f "$dest" ]; then
    # Break hardlink if present (nlink > 1)
    nlink=$(stat -c '%h' "$dest" 2>/dev/null || echo 1)
    if [ "$nlink" -gt 1 ]; then
      echo "[ensure] breaking hardlink: $dest"
      tmp="${dest}.tmp.$$"
      cp -a -- "$dest" "$tmp"
      mv -f -- "$tmp" "$dest"
    else
      echo "[ensure] ok: $dest"
    fi
  else
    src="${SRC[$name]}"
    if [ ! -f "$src" ]; then
      echo "[ensure] MISSING weight for ${name}."
      echo "  Expected: $dest"
      echo "  Set PRETRAINED_BUNDLE to a dir containing ${name}/pytorch_model.bin."
      exit 1
    fi
    echo "[ensure] copying $src -> $dest"
    cp -a -- "$src" "$dest"
  fi
  ln -sfn pytorch_model.bin "${dest_dir}/checkpoint.pth"
done

echo "[ensure] done."
