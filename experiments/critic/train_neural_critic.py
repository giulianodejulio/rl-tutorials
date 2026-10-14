"""First neural Monte Carlo critic: fixed policy, saved data, full-batch SGD."""

from pathlib import Path

import numpy as np
import torch
from torch import nn


def build_critic(hidden_size):
    # Two inputs (e, h), learned hidden features, unrestricted scalar output.
    return nn.Sequential(nn.Linear(2, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))


def train_critic(model, x, y, validation_x, validation_y, learning_rate, num_epochs):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    train_history, validation_history = [], []
    for epoch in range(num_epochs + 1):
        predictions = model(x).squeeze(-1)
        mse = torch.mean((predictions - y)**2)
        with torch.no_grad():
            validation_mse = torch.mean((model(validation_x).squeeze(-1) - validation_y)**2)
        train_history.append(mse.item())
        validation_history.append(validation_mse.item())
        if not np.isfinite(train_history[-1]) or not np.isfinite(validation_history[-1]):
            raise FloatingPointError("Non-finite critic MSE")
        if epoch % 500 == 0 or epoch == num_epochs:
            print(f"epoch={epoch}: train_mse={mse.item():.6f}, "
                  f"validation_mse={validation_mse.item():.6f}", flush=True)
        if epoch == num_epochs:
            break
        optimizer.zero_grad()
        loss = mse / 2
        loss.backward()
        optimizer.step()
    return np.array(train_history), np.array(validation_history)


def main():
    directory = Path(__file__).resolve().parents[2] / "results" / "critic" / "data"
    with np.load(directory / "critic_dataset.npz", allow_pickle=False) as data:
        # Select e and h; e^2 is not supplied as a handcrafted feature.
        train_inputs = data["train_features"][..., [1, 3]]
        validation_inputs = data["validation_features"][..., [1, 3]]
        train_returns = data["train_returns"]
        validation_returns = data["validation_returns"]
        metadata = {key: data[key] for key in (
            "policy_weights", "sigma", "target_velocity", "num_steps",
            "training_seed", "validation_seed",
        )}
    x = torch.tensor(train_inputs.reshape(-1, 2), dtype=torch.float32)
    y = torch.tensor(train_returns.reshape(-1), dtype=torch.float32)
    validation_x = torch.tensor(validation_inputs.reshape(-1, 2), dtype=torch.float32)
    validation_y = torch.tensor(validation_returns.reshape(-1), dtype=torch.float32)
    initialization_seed = 123
    torch.manual_seed(initialization_seed)
    torch.set_num_threads(1)
    hidden_size, learning_rate, num_epochs = 16, 0.05, 3000
    model = build_critic(hidden_size)
    train_mse, validation_mse = train_critic(
        model, x, y, validation_x, validation_y, learning_rate, num_epochs
    )
    with torch.no_grad():
        validation_predictions = model(validation_x).squeeze(-1).numpy().reshape(validation_returns.shape)

    # Refit the same linear reference on this dataset for a consistent comparison.
    linear_x = np.column_stack((np.ones(len(y)), x.numpy()[:, 0],
                                x.numpy()[:, 0]**2, x.numpy()[:, 1]))
    theta_ls = np.linalg.lstsq(linear_x, train_returns.reshape(-1), rcond=None)[0]
    val_flat = validation_inputs.reshape(-1, 2)
    linear_val_x = np.column_stack((np.ones(len(val_flat)), val_flat[:, 0],
                                    val_flat[:, 0]**2, val_flat[:, 1]))
    ls_predictions = (linear_val_x @ theta_ls).reshape(validation_returns.shape)
    baseline = train_returns.mean(axis=0)
    print(f"Neural validation MSE: {validation_mse[-1]:.6f}")
    print(f"Linear least-squares validation MSE: {np.mean((validation_returns-ls_predictions)**2):.6f}")
    print(f"Temporal baseline validation MSE: {np.mean((validation_returns-baseline)**2):.6f}")
    model_path = directory / "neural_critic.pt"
    torch.save(dict(state_dict=model.state_dict(), hidden_size=hidden_size,
                    input_names=["velocity_error", "remaining_fraction"]), model_path)
    path = directory / "neural_critic.npz"
    np.savez(path, train_mse=train_mse, validation_mse=validation_mse,
             validation_predictions=validation_predictions,
             ls_validation_predictions=ls_predictions, theta_ls=theta_ls,
             ls_train_mse=np.mean((train_returns.reshape(-1)-linear_x @ theta_ls)**2),
             ls_validation_mse=np.mean((validation_returns-ls_predictions)**2),
             time_baseline=baseline,
             baseline_validation_mse=np.mean((validation_returns-baseline)**2),
             hidden_size=hidden_size, learning_rate=learning_rate, num_epochs=num_epochs,
             initialization_seed=initialization_seed,
             input_names=np.array(["velocity_error", "remaining_fraction"]), **metadata)
    print(f"Saved: {path}\nSaved: {model_path}")


if __name__ == "__main__":
    main()
