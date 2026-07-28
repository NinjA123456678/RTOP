"""Load and inspect a synthetic best-validation checkpoint."""

import argparse

import torch

from rtqp_phase1.training import BestCheckpointManager


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    model = torch.nn.Linear(1, 1)
    payload = BestCheckpointManager(args.checkpoint).load_best(model)
    print(
        f"SYNTHETIC SMOKE loaded best validation checkpoint: "
        f"epoch={payload['epoch']} metric={payload['validation_metric']}"
    )


if __name__ == "__main__":
    main()
