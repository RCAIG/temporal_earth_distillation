#!/usr/bin/env bash
# Ensure pretrained/*/pytorch_model.bin exist as real files (not hardlinks).
# Copies from a local best-epoch bundle if weights are missing.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUNDLE="${PRETRAINED_BUNDLE:-/work/projects/resilientia/ziyun/TimeSeries_SSL_USA/reports/fullscale_12b768_best_ep_glance_cropharvest_checkpoint_bundle}"

declare -A SRC=(
  [ted-hls-12b768]="${BUNDLE}/job9616_TED_cXattnB_ep72/checkpoint_epoch_72.pth"
  [msm-hls-12b768]="${BUNDLE}/job10909_MSM_reg4_ep25/checkpoint_epoch_25.pth"
  [ntp-hls-12b768]="${BUNDLE}/job10685_NTP_ep25/checkpoint_epoch_25.pth"
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
      echo "  Or set PRETRAINED_BUNDLE to a dir containing the best-epoch .pth files."
      exit 1
    fi
    echo "[ensure] copying $src -> $dest"
    cp -a -- "$src" "$dest"
  fi
  ln -sfn pytorch_model.bin "${dest_dir}/checkpoint.pth"
done

echo "[ensure] done."
