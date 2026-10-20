"""Full two-action REINFORCE training, reusing the single-update experiment."""

from pathlib import Path

import numpy as np
import torch

from experiments.policy_gradient.reinforce_neural_2d import build_policy, collect_batch, update_policy


def main():
    torch.manual_seed(123)
    torch.set_num_threads(1)
    hidden_size, num_steps, num_updates = 16, 200, 50
    num_rollouts, num_reference_rollouts, num_evaluation_rollouts = 100, 100, 200
    target_velocity, sigma = np.array([0.8, 0.4]), np.array([0.3, 0.3])
    learning_rate = 0.00001
    model = build_policy(hidden_size)
    rng, reference_rng = np.random.default_rng(42), np.random.default_rng(31415)
    evaluation_updates, evaluation_returns, evaluation_velocities = [], [], []

    def evaluate(m):
        batch = collect_batch(model, target_velocity, sigma, num_steps,
                              num_evaluation_rollouts, np.random.default_rng(2026))
        evaluation_updates.append(m)
        evaluation_returns.append(batch["returns"][:, 0])
        evaluation_velocities.append(batch["inputs"][:, :, 2:4])
        print(f"update={m}: evaluation={evaluation_returns[-1].mean():.3f}", flush=True)

    evaluate(0)
    weights = [torch.cat([p.detach().reshape(-1) for p in model.parameters()]).numpy().copy()]
    losses, norms, batch_returns, baselines = [], [], [], []
    for m in range(num_updates):
        reference = collect_batch(model, target_velocity, sigma, num_steps,
                                  num_reference_rollouts, reference_rng)
        baseline = reference["returns"].mean(axis=0)
        batch = collect_batch(model, target_velocity, sigma, num_steps, num_rollouts, rng)
        loss, gradient = update_policy(model, batch, baseline, sigma, learning_rate)
        losses.append(loss)
        norms.append(np.linalg.norm(gradient))
        batch_returns.append(batch["returns"][:, 0])
        baselines.append(baseline)
        weights.append(torch.cat([p.detach().reshape(-1) for p in model.parameters()]).numpy().copy())
        if (m + 1) % 10 == 0 or m + 1 == num_updates:
            evaluate(m + 1)
    directory = Path(__file__).resolve().parents[2] / "results" / "policy_gradient" / "data"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "reinforce_neural_2d_training.npz"
    np.savez(path, weights=np.array(weights), policy_losses=np.array(losses),
             gradient_norms=np.array(norms), batch_returns=np.array(batch_returns),
             time_baselines=np.array(baselines), evaluation_updates=np.array(evaluation_updates),
             evaluation_returns=np.array(evaluation_returns),
             evaluation_velocities=np.array(evaluation_velocities),
             target_velocity=target_velocity, sigma=sigma, learning_rate=learning_rate,
             hidden_size=hidden_size, num_steps=num_steps, num_updates=num_updates,
             num_rollouts=num_rollouts, num_reference_rollouts=num_reference_rollouts,
             num_evaluation_rollouts=num_evaluation_rollouts, initialization_seed=123,
             training_seed=42, reference_seed=31415, evaluation_seed=2026)
    torch.save(dict(state_dict=model.state_dict(), hidden_size=hidden_size,
                    sigma=sigma.tolist()), directory / "reinforce_neural_2d_trained.pt")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
