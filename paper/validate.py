"""Integration checks without learning, gradients or optimizer updates."""
import copy
import json
from pathlib import Path
import random
import shutil

import mujoco
import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from paper.artifacts import prepare_run, resolve_checkpoint, save_checkpoint
from paper.config import ROOT, curriculum, load_config
from paper.env import PaperSolo12Env, pd_torque, power_loss
from paper.train import TrainingHistory, build_model, make_env


def main():
    torch.set_num_threads(1)
    c = load_config()
    env = PaperSolo12Env(c, randomize=False, reward_scale=.4, noise_scale=.2)
    check_env(env, warn=True)
    obs, _ = env.reset(seed=7, options={'command': [.2, 0, 0]})
    assert obs.shape == (132,)
    assert np.isfinite(obs).all()
    np.testing.assert_allclose(obs[:3], env.data.xmat[env.base].reshape(3,3)[2])
    old_error = env.error_history.copy()
    old_target = env.last_target.copy()
    action = np.ones(12)*.2
    obs, reward, term, trunc, info = env.step(action)
    target = env.nominal+.3*action
    np.testing.assert_allclose(env.last_target, target)
    np.testing.assert_allclose(env.error_history[0], target-env.data.qpos[env.qadr])
    np.testing.assert_allclose(env.error_history[1], old_error[0])
    np.testing.assert_allclose(obs[36:48], target)
    np.testing.assert_allclose(obs[48:60], old_target)
    np.testing.assert_allclose(info['cost_smoothness1'], 2.5*np.sum((target-old_target)**2))
    np.testing.assert_allclose(info['cost_smoothness2'], 1.5*np.sum((target-old_target)**2))
    np.testing.assert_allclose(reward, info['tracking']-.4*info['penalty'])
    np.testing.assert_allclose(info['full_reward'], info['tracking']-info['penalty'])
    assert np.max(np.abs(env.data.actuator_force)) <= 3+1e-12
    np.testing.assert_allclose(pd_torque(np.array([10., -10.]), np.zeros(2), np.zeros(2),3,.2,3), [3,-3])
    np.testing.assert_allclose(power_loss(np.ones(12),np.zeros(12)),12/4.81)
    assert power_loss(np.zeros(12),np.ones(12)) > 0
    env.data.qvel[:] = 0
    env.data.qpos[3:7] = [np.cos(np.pi/4),0,0,np.sin(np.pi/4)]
    env.data.qvel[0] = 1
    mujoco.mj_forward(env.model,env.data)
    _, velocity, _ = env._body_state()
    np.testing.assert_allclose(velocity, [0,-1,0], atol=1e-10)
    env.close()

    small = copy.deepcopy(c)
    small.update(episode_steps=2, continuation_probability=1., num_envs=2, buffer_size=200)
    env = PaperSolo12Env(small, randomize=True, reward_scale=1., noise_scale=1., rough_enabled=True)
    first, _ = env.reset(seed=11, options={'rough': True})
    terrain = env.heights.copy()
    again, _ = env.reset(seed=11, options={'rough': True})
    np.testing.assert_array_equal(first,again)
    np.testing.assert_array_equal(terrain,env.heights)
    assert terrain.max() <= .05 and terrain.min() >= .001
    # Test truncation/continuation separately without rough-ground failure.
    env.randomize = False
    env.reset(seed=5, options={'rough':False,'command':[0,0,0]})
    _,_,term,trunc,_ = env.step(np.zeros(12))
    assert not term and not trunc
    _,_,term,trunc,_ = env.step(np.zeros(12))
    assert trunc and not term
    qpos = env.data.qpos.copy()
    _, reset_info = env.reset(options={'command':[.2,0,0]})
    assert not reset_info['hard_reset']
    np.testing.assert_array_equal(qpos,env.data.qpos)
    # Force a non-foot collision; this must be true termination, not timeout.
    env.data.qpos[2] -= .25
    mujoco.mj_forward(env.model,env.data)
    _,_,term,trunc,info = env.step(np.zeros(12))
    assert term and not trunc and info['fallen']
    np.testing.assert_allclose(info['full_reward'],info['tracking']-info['penalty']-10)
    env.close()
    np.testing.assert_allclose(curriculum(c,0)[:2], [0,0])
    np.testing.assert_allclose(curriculum(c,150000000)[:2],[1,1])
    assert not curriculum(c,269970000)[2] and curriculum(c,270000000)[2]

    vec = SubprocVecEnv([make_env(small,j) for j in range(2)], start_method='spawn')
    vec.reset()
    vec.env_method('set_curriculum', .3,.2,False)
    for _ in range(2):
        obs, reward, done, infos = vec.step(np.zeros((2,12)))
    assert done.all()
    assert all(i['TimeLimit.truncated'] and i['terminal_observation'].shape==(132,) for i in infos)
    vec.close()

    # Untrained checkpoint fixture. No model.learn / train / optimizer calls.
    path = ROOT/'checks'/'interface'
    if path.exists():
        shutil.rmtree(path)
    prepare_run(path,small)
    vec = VecNormalize(DummyVecEnv([make_env(small,j) for j in range(2)]), norm_reward=False)
    model = build_model(vec,small)
    assert model.policy.log_std.requires_grad
    assert not model.policy.optimizer.state
    callback = TrainingHistory(c,path)
    callback.init_callback(model)
    for n, expected_reward, expected_noise, rough in ((0,0.,0.,False),
            (150000000,1.,1.,False),(270000000,1.,1.,True)):
        callback.num_timesteps = n
        callback.configure_phase()
        assert np.allclose(vec.env_method('get_wrapper_attr','reward_scale'), expected_reward)
        assert np.allclose(vec.env_method('get_wrapper_attr','noise_scale'), expected_noise)
        assert all(value == rough for value in vec.env_method('get_wrapper_attr','rough_enabled'))
        assert model.ent_coef == (0. if rough else .01)
        assert model.clip_range(1.) == (.05 if rough else .2)
    callback.num_timesteps = 0
    callback.configure_phase()
    vec.reset()
    for _ in range(2):
        vec.step(np.zeros((2,12)))
    snapshot = [p.detach().clone() for p in model.policy.parameters()]
    checkpoint = save_checkpoint(model,vec,path,trained=False)
    resolve_checkpoint(path)
    # Exercise the exact training callback checkpoint/evaluation path at zero
    # updates, without calling PPO.learn or PPO.train.
    callback._on_rollout_start()
    from paper.evaluate import evaluate_run
    torch_state, numpy_state, python_state = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    rows = evaluate_run(path,checkpoint,episodes=1,duration=.08,gifs=True,
                        selected_commands=['x02'],randomized=False)
    assert not model.policy.optimizer.state
    assert len(rows)==3 and all(not p.requires_grad or p.grad is None for p in model.policy.parameters())
    assert torch.equal(torch.get_rng_state(),torch_state)
    assert np.array_equal(np.random.get_state()[1],numpy_state[1]) and random.getstate()==python_state
    for old,new in zip(snapshot,model.policy.parameters()):
        assert torch.equal(old,new)
    vec.close()
    report = {'checks':'passed','training_executed':False,'optimization_steps':0,
              'observation_size':132,'physics_dt':c['physics_dt'],'control_dt':c['control_dt']}
    (ROOT/'checks'/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
