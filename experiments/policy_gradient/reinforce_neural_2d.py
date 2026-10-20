"""First vector-action experiment: one REINFORCE update in PointMass2D."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.continuous_control.point_mass_2d import PointMass2D
from experiments.policy_gradient.reinforce_batch_gradient import returns_to_go


def build_policy(hidden_size):
    return nn.Sequential(nn.Linear(6, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 2))


def collect_batch(model, target_velocity, sigma, num_steps, num_rollouts, rng):
    env = PointMass2D()
    inputs = np.zeros((num_rollouts, num_steps, 6))
    means = np.zeros((num_rollouts, num_steps, 2))
    actions = np.zeros_like(means)
    rewards = np.zeros((num_rollouts, num_steps))
    returns = np.zeros_like(rewards)
    for i in range(num_rollouts):
        observation = env.reset()
        trajectory = []
        for t in range(num_steps):
            inputs[i, t] = np.concatenate((observation, target_velocity))
            with torch.no_grad():
                means[i, t] = model(torch.tensor(inputs[i, t], dtype=torch.float32)).numpy()
            # Independent noise components conditional on the current state.
            actions[i, t] = rng.normal(means[i, t], sigma)
            observation, rewards[i, t] = env.step(actions[i, t], target_velocity)
            trajectory.append(dict(reward=rewards[i, t]))
        returns[i] = returns_to_go(trajectory)
    if not all(np.isfinite(a).all() for a in (inputs, means, actions, returns)):
        raise FloatingPointError("Non-finite rollout")
    return dict(inputs=inputs, means=means, actions=actions, rewards=rewards, returns=returns)


def update_policy(model, batch, baseline, sigma, learning_rate):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    x = torch.tensor(batch["inputs"], dtype=torch.float32)
    a = torch.tensor(batch["actions"], dtype=torch.float32)
    advantages = torch.tensor(batch["returns"] - baseline, dtype=torch.float32)
    means = model(x)
    distribution = torch.distributions.Normal(means, torch.tensor(sigma, dtype=torch.float32))
    component_log_probs = distribution.log_prob(a)  # (N,T,2)
    joint_log_probs = component_log_probs.sum(dim=-1)  # (N,T)
    loss = -(advantages * joint_log_probs).sum(dim=1).mean()
    optimizer.zero_grad()
    loss.backward()
    gradient = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Non-finite gradient")
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("Non-finite updated weights")
    return loss.item(), gradient.detach().numpy()


def main():
    torch.manual_seed(123)
    torch.set_num_threads(1)
    hidden_size, num_steps = 16, 200
    num_rollouts, num_reference_rollouts, num_evaluation_rollouts = 100, 100, 200
    target_velocity = np.array([0.8, 0.4])
    sigma = np.array([0.3, 0.3])
    learning_rate = 0.00001
    model = build_policy(hidden_size)
    initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    evaluation_before = collect_batch(model, target_velocity, sigma, num_steps,
                                      num_evaluation_rollouts, np.random.default_rng(2026))
    reference = collect_batch(model, target_velocity, sigma, num_steps,
                              num_reference_rollouts, np.random.default_rng(31415))
    baseline = reference["returns"].mean(axis=0)
    batch = collect_batch(model, target_velocity, sigma, num_steps,
                          num_rollouts, np.random.default_rng(42))
    loss, gradient = update_policy(model, batch, baseline, sigma, learning_rate)
    evaluation_after = collect_batch(model, target_velocity, sigma, num_steps,
                                     num_evaluation_rollouts, np.random.default_rng(2026))
    # At the initial state, every rollout has the same inputs.
    print(f"Initial mean at reset: {evaluation_before['means'][0, 0]}")
    print(f"Updated mean at reset: {evaluation_after['means'][0, 0]}")
    before = evaluation_before["returns"][:, 0]
    after = evaluation_after["returns"][:, 0]
    print(f"Return before: {before.mean():.3f}; after one update: {after.mean():.3f}")
    print(f"Paired improvement: {(after-before).mean():.3f}")
    print(f"Gradient norm: {np.linalg.norm(gradient):.3f}")
    directory = Path(__file__).resolve().parents[2] / "results" / "policy_gradient" / "data"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "reinforce_neural_2d.npz"
    arrays = {f"{prefix}_{key}": value
              for prefix, data in (("batch", batch), ("evaluation_before", evaluation_before),
                                   ("evaluation_after", evaluation_after))
              for key, value in data.items()}
    np.savez(path, **arrays, time_baseline=baseline, gradient=gradient, policy_loss=loss,
             sigma=sigma, target_velocity=target_velocity, learning_rate=learning_rate,
             hidden_size=hidden_size, num_steps=num_steps, num_rollouts=num_rollouts,
             num_reference_rollouts=num_reference_rollouts,
             num_evaluation_rollouts=num_evaluation_rollouts,
             initialization_seed=123, training_seed=42, reference_seed=31415,
             evaluation_seed=2026)
    torch.save(dict(initial_state_dict=initial_state, state_dict=model.state_dict(),
                    hidden_size=hidden_size, sigma=sigma.tolist()), directory / "reinforce_neural_2d.pt")
    print("One update, one seed; original Gaussian actions retained before clipping.")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
