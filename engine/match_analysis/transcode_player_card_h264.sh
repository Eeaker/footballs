#!/usr/bin/env bash
set -euo pipefail

source_dir="${1:?source package is required}"
target_dir="${2:?target package is required}"
ffmpeg_bin="${FFMPEG_BIN:?FFMPEG_BIN is required}"

if [[ -e "$target_dir" ]]; then
  echo "target already exists: $target_dir" >&2
  exit 2
fi

cp -al -- "$source_dir" "$target_dir"
export ffmpeg_bin
export MAX_WIDTH="${MAX_WIDTH:-1280}"
export H264_CRF="${H264_CRF:-28}"

transcode_one() {
  local output="$1"
  local temporary="${output%.mp4}.h264tmp.mp4"
  local scale_filter="scale='min(${MAX_WIDTH:-1280},iw)':-2"
  "$ffmpeg_bin" -loglevel error -y -i "$output" \
    -map 0:v:0 -vf "$scale_filter" -c:v libx264 -preset veryfast \
    -crf "${H264_CRF:-28}" \
    -pix_fmt yuv420p -movflags +faststart -an "$temporary"
  mv -- "$temporary" "$output"
}
export -f transcode_one

find "$target_dir" -type f -name '*.mp4' -print0 \
  | xargs -0 -P "${TRANSCODE_JOBS:-8}" -I '{}' bash -c 'transcode_one "$1"' _ '{}'

printf 'complete\n' > "${target_dir}.transcode.done"
