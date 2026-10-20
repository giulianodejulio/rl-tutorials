#!/usr/bin/env bash
set -euo pipefail
setup_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$setup_dir/.." && pwd)"
if [[ "$(basename -- "$(dirname -- "$repo_dir")")" != src ]]; then
  echo 'Clone into ~/workspaces/solo12_rl_ws/src/rl-tutorials first.' >&2
  exit 1
fi
workspace_dir="$(cd -- "$repo_dir/../.." && pwd)"
python3 - "$repo_dir" "$workspace_dir" <<'PY'
import hashlib,json,shlex,shutil,subprocess,sys
from pathlib import Path
repo,workspace=map(Path,sys.argv[1:])
manifest=json.loads((repo/'workspace/model/manifest.json').read_text())
model=workspace/'third_party/solo12_mujoco'
model.parent.mkdir(parents=True,exist_ok=True)
if not model.exists():
    subprocess.run(['git','clone',manifest['repository'],str(model)],check=True)
    subprocess.run(['git','-C',str(model),'checkout','--detach',manifest['revision']],check=True)
revision=subprocess.check_output(['git','-C',str(model),'rev-parse','HEAD'],text=True).strip()
if revision!=manifest['revision']:
    raise SystemExit(f'Existing model has different HEAD: {revision}. No files changed; use a separate workspace.')
source=repo/'workspace/model/solo12_gen.xml'
if hashlib.sha256(source.read_bytes()).hexdigest()!=manifest['generated_xml_sha256']:
    raise SystemExit('Versioned generated model does not match its manifest')
target=model/'resources/solo12_gen.xml'
if target.exists() and target.read_bytes()!=source.read_bytes():
    raise SystemExit(f'Existing {target} differs; refusing to overwrite it')
if not target.exists():
    shutil.copy2(source,target)
digest=hashlib.sha256()
for path in sorted(model.rglob('*')):
    if path.is_file() and path.suffix in ('.xml','.obj','.stl'):
        digest.update(str(path.relative_to(model)).encode())
        digest.update(path.read_bytes())
if digest.hexdigest()!=manifest['assets_sha256']:
    raise SystemExit('Shared model assets differ from manifest; inspect local changes')
runtime=workspace/'docker'
runtime.mkdir(exist_ok=True)
(runtime/'.bash_history').touch(exist_ok=True)
params=runtime/'params.env'
if not params.exists():
    params.write_text('export WORKSPACE_TO_SOURCE='+shlex.quote(str(workspace))+'\n'
                      'export SOLO12_RL_WS='+shlex.quote(str(workspace))+'\n')
print(f'Workspace: {workspace}')
print(f'Model revision and assets verified: {revision}')
print('Next: bash workspace/build.sh; bash workspace/run.sh')
PY
