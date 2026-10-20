"""Configuration, artifact paths and provenance; no imports from tutorial envs."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCENE = ROOT.parents[2] / 'third_party' / 'solo12_mujoco' / 'scene.xml'


def load_config(path=None):
    config = json.loads(Path(path or ROOT/'configs'/'article.json').read_text())
    if config['buffer_size'] % config['num_envs']:
        raise ValueError('buffer_size must be divisible by num_envs')
    if config['buffer_size'] % config['minibatches']:
        raise ValueError('buffer_size must be divisible by minibatches')
    return config


def curriculum(config, timesteps):
    # Canonical iteration = 300 robots * 100 steps, independent of CPU workers.
    iteration = timesteps / config['buffer_size']
    return (min(1., iteration*config['reward_increment']),
            min(1., iteration*config['noise_increment']),
            iteration >= config['terrain_start_iteration'])


@lru_cache(maxsize=1)
def model_digest():
    digest = hashlib.sha256()
    for path in sorted(SCENE.parent.rglob('*')):
        if path.is_file() and path.suffix in ('.xml', '.obj', '.stl'):
            digest.update(str(path.relative_to(SCENE.parent)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def run_path(name):
    path = ROOT/'results'/name
    if not path.resolve().is_relative_to((ROOT/'results').resolve()):
        raise ValueError('Run name must stay inside paper/results')
    return path
