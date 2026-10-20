"""Neural REINFORCE with a separate temporal baseline; one update per batch."""

from pathlib import Path

import numpy as np
import torch

from experiments.continuous_control.continuous_control_rollout import PointMass1D
from experiments.policy_gradient.neural_gaussian_policy import build_policy, collect_neural_rollout
from experiments.policy_gradient.reinforce_batch_gradient import returns_to_go


def collect_batch(model, sigma, target_velocity, num_steps, num_rollouts, rng):
    env = PointMass1D()
    trajectories = [collect_neural_rollout(
        env, model, sigma, target_velocity, num_steps, rng
    ) for _ in range(num_rollouts)]
    inputs = np.array([[s["inputs"] for s in tau] for tau in trajectories])
    actions = np.array([[s["action"] for s in tau] for tau in trajectories])
    returns = np.array([returns_to_go(tau) for tau in trajectories])
    if not all(np.isfinite(a).all() for a in (inputs, actions, returns)):
        raise FloatingPointError("Non-finite rollout data")
    return inputs, actions, returns


def update_policy(model, optimizer, inputs, actions, returns, baseline, sigma):
    # Recorded states, actions and Monte Carlo targets are fixed data.
    x = torch.tensor(inputs, dtype=torch.float32)
    a = torch.tensor(actions, dtype=torch.float32)
    advantages = torch.tensor(returns - baseline, dtype=torch.float32)
    means = model(x).squeeze(-1)
    distribution = torch.distributions.Normal(means, sigma)
    log_probabilities = distribution.log_prob(a)
    # Sum over time, average over rollouts: do not average over N*T.
    loss = -(advantages * log_probabilities).sum(dim=1).mean()
    optimizer.zero_grad()
    loss.backward()
    gradients = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    if not torch.isfinite(gradients).all():
        raise FloatingPointError("Non-finite policy gradient")
    norm = gradients.norm().item()
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("Non-finite policy weights")
    return loss.item(), norm


def evaluate_neural_policy(model, sigma, target_velocity, num_steps, num_rollouts):
    # Same noise across checkpoints, separate from training/reference noise.
    env = PointMass1D()
    rng = np.random.default_rng(2026)
    return np.array([sum(s["reward"] for s in collect_neural_rollout(
        env, model, sigma, target_velocity, num_steps, rng
    )) for _ in range(num_rollouts)])


def train(model, sigma, target_velocity, num_steps, num_rollouts,
          num_reference_rollouts, num_updates, learning_rate, num_evaluation_rollouts):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    rng, reference_rng = np.random.default_rng(42), np.random.default_rng(31415)
    evaluation_updates = [0]
    evaluations = [evaluate_neural_policy(
        model, sigma, target_velocity, num_steps, num_evaluation_rollouts
    )]
    weights = [torch.cat([p.detach().reshape(-1) for p in model.parameters()]).numpy().copy()]
    losses, norms, batch_returns, baselines = [], [], [], []
    for m in range(num_updates):
        _, _, reference_returns = collect_batch(
            model, sigma, target_velocity, num_steps, num_reference_rollouts, reference_rng
        )
        baseline = reference_returns.mean(axis=0)
        inputs, actions, returns = collect_batch(
            model, sigma, target_velocity, num_steps, num_rollouts, rng
        )
        loss, norm = update_policy(model, optimizer, inputs, actions, returns, baseline, sigma)
        losses.append(loss)
        norms.append(norm)
        batch_returns.append(returns[:, 0])
        baselines.append(baseline)
        weights.append(torch.cat([p.detach().reshape(-1) for p in model.parameters()]).numpy().copy())
        if m == 0 or (m + 1) % 10 == 0:
            print(f"update={m + 1}: batch_return_before_update={returns[:, 0].mean():.3f}, "
                  f"gradient_norm={norm:.3f}", flush=True)
        if (m + 1) % 10 == 0 or m + 1 == num_updates:
            evaluation_updates.append(m + 1)
            evaluations.append(evaluate_neural_policy(
                model, sigma, target_velocity, num_steps, num_evaluation_rollouts
            ))
            print(f"  evaluation_after_update={evaluations[-1].mean():.3f}", flush=True)
    return dict(weights=np.array(weights), policy_losses=np.array(losses),
                gradient_norms=np.array(norms), batch_returns=np.array(batch_returns),
                time_baselines=np.array(baselines),
                evaluation_updates=np.array(evaluation_updates),
                evaluation_returns=np.array(evaluations))


def main():
    hidden_size, initialization_seed = 16, 123
    torch.manual_seed(initialization_seed)
    torch.set_num_threads(1)
    model = build_policy(hidden_size)
    configuration = dict(sigma=0.3, target_velocity=0.8, num_steps=200,
                         num_rollouts=100, num_reference_rollouts=100,
                         num_updates=50, learning_rate=0.00001, num_evaluation_rollouts=200)
    history = train(model, **configuration)
    directory = Path(__file__).resolve().parents[2] / "results" / "policy_gradient" / "data"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "neural_gaussian_policy_training.npz"
    np.savez(path, **history, **configuration, hidden_size=hidden_size,
             initialization_seed=initialization_seed, training_seed=42,
             reference_seed=31415, evaluation_seed=2026)
    torch.save(dict(state_dict=model.state_dict(), hidden_size=hidden_size,
                    input_names=["position", "velocity", "target_velocity"],
                    sigma=configuration["sigma"]), directory / "neural_gaussian_policy_trained.pt")
    before, after = history["evaluation_returns"][[0, -1]]
    print(f"Initial return: {before.mean():.3f}; final return: {after.mean():.3f}")
    print(f"Paired mean improvement: {(after-before).mean():.3f}")
    print("One training seed. Surrogate policy loss is not a prediction MSE.")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
