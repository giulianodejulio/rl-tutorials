#!/usr/bin/env bash
set -euo pipefail
setup_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$setup_dir/.." && pwd)"
workspace_dir="$(cd -- "$repo_dir/../.." && pwd)"
image="${SOLO12_IMAGE:-solo12_rl_ws:base}"
docker build --tag "$image" \
  --build-arg USERNAME="$(id -un)" \
  --build-arg USERID="$(id -u)" \
  --build-arg GROUPID="$(id -g)" \
  --build-arg WORKSPACE="$workspace_dir" \
  --build-arg WORKSPACE_NAME="$(basename -- "$workspace_dir")" \
  --build-arg TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}" \
  --file "$setup_dir/Dockerfile" "$@" "$setup_dir"
