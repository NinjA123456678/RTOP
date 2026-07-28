"""Synthetic smoke run for the Phase 1 trainer/checkpoint lifecycle.

This does not train RTQP and does not touch the legacy entry point.
"""

import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset

from rtqp_phase1.config import TrainingConfig
from rtqp_phase1.training import ValidationCheckpointTrainer, masked_mae, masked_rmse


def loaders(batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    x = torch.linspace(0.0, 1.0, 32).unsqueeze(1)
    y = 2.0 * x.squeeze(1) + 0.5
    dataset = TensorDataset(x, y)
    return tuple(DataLoader(dataset, batch_size=batch_size, shuffle=False) for _ in range(3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    config = TrainingConfig(batch_size=8, max_epochs=args.epochs, selection_metric="val_loss")
    train_loader, validation_loader, test_loader = loaders(config.batch_size)
    model = torch.nn.Linear(1, 1)

    def training_loss(current_model, batch):
        x, target = batch
        return torch.mean((current_model(x).squeeze(-1) - target) ** 2)

    def evaluate(current_model, loader):
        current_model.eval()
        predictions, targets = [], []
        with torch.no_grad():
            for x, target in loader:
                predictions.append(current_model(x).squeeze(-1))
                targets.append(target)
        prediction, target = torch.cat(predictions), torch.cat(targets)
        mask = torch.ones_like(target, dtype=torch.bool)
        return {
            "val_loss": float(torch.mean((prediction - target) ** 2)),
            "mae": float(masked_mae(prediction, target, mask)),
            "rmse": float(masked_rmse(prediction, target, mask)),
        }

    result = ValidationCheckpointTrainer(
        config, checkpoint_path=args.checkpoint
    ).fit_and_test(
        model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        training_loss=training_loss,
        evaluate=evaluate,
    )
    print(f"SYNTHETIC SMOKE best_epoch={result['best_epoch']} test={result['test_metrics']}")


if __name__ == "__main__":
    main()
