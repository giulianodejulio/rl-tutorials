"""Paired model/normalizer checkpoints and reproducible run manifests."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from paper.config import ROOT, SCENE, model_digest


def env_digest():
    return hashlib.sha256((ROOT/'env.py').read_bytes()+(ROOT/'config.py').read_bytes()).hexdigest()


def prepare_run(directory, config):
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    source = directory/'source'
    source.mkdir()
    for path in list(ROOT.glob('*.py')) + [ROOT/'SOURCES.md', ROOT/'README.md']:
        if path.exists():
            shutil.copy2(path, source/path.name)


def save_checkpoint(model, normalizer, directory, trained=True):
    import gymnasium
    import mujoco
    import stable_baselines3
    import torch
    parent = directory/'checkpoints'
    parent.mkdir(exist_ok=True)
    destination = parent/f'{model.num_timesteps:012d}'
    if not destination.exists():
        stage = Path(tempfile.mkdtemp(dir=parent, prefix='.tmp-'))
        try:
            model.save(stage/'policy.zip')
            normalizer.save(stage/'normalization.pkl')
            meta = dict(timesteps=model.num_timesteps, trained=trained,
                        env_sha256=env_digest(), model_sha256=model_digest(), shared_scene=str(SCENE),
                        mujoco=mujoco.__version__, torch=torch.__version__,
                        sb3=stable_baselines3.__version__, gymnasium=gymnasium.__version__)
            (stage/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
            os.replace(stage, destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    pointer = directory/'latest.tmp'
    pointer.write_text(json.dumps({'checkpoint': str(destination.relative_to(directory))})+'\n')
    os.replace(pointer, directory/'latest.json')
    return destination


def resolve_checkpoint(directory, step=None):
    if step is not None:
        path = directory/'checkpoints'/f'{step:012d}'
    else:
        path = directory/json.loads((directory/'latest.json').read_text())['checkpoint']
    metadata = json.loads((path/'metadata.json').read_text())
    if metadata['env_sha256'] != env_digest():
        raise ValueError('Environment code differs from checkpoint; use run/source to restore it')
    if metadata['model_sha256'] != model_digest():
        raise ValueError('Shared MuJoCo assets differ from the saved model manifest')
    return path, metadata
