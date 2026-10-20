#!/usr/bin/env bash
set -euo pipefail
setup_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$setup_dir/.." && pwd)"
workspace_dir="$(cd -- "$repo_dir/../.." && pwd)"
container="${SOLO12_CONTAINER:-docker_solo12_rl_ws}"
image="${SOLO12_IMAGE:-solo12_rl_ws:base}"
options=()
while (($#)); do
  case "$1" in
    --gpu) options+=(--gpus all -e MUJOCO_GL=egl -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics);;
    *) echo "Unknown option: $1" >&2; exit 1;;
  esac
  shift
done
if docker container inspect "$container" >/dev/null 2>&1; then
  echo "Container already exists: $container. Use docker start/exec, or another SOLO12_CONTAINER." >&2
  exit 1
fi
docker run -d --name "$container" --init --shm-size=1g \
  --mount "type=bind,src=$workspace_dir,dst=$workspace_dir" \
  --workdir "$repo_dir" \
  "${options[@]}" "$image" sleep infinity
printf 'Ready. Enter with: docker exec -it %q bash\n' "$container"
