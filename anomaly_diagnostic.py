"""Export normal/anomalous score and confidence summaries from NPZ arrays."""

import argparse
import json

import numpy as np

from rtqp_phase1.experiments import anomaly_diagnostics, export_anomaly_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="NPZ with scores/confidence/anomaly_mask/observed_mask")
    parser.add_argument("--output", required=True, help="Destination CSV")
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as payload:
        result = anomaly_diagnostics(
            payload["scores"],
            payload["confidence"],
            payload["anomaly_mask"],
            payload["observed_mask"],
        )
    export_anomaly_diagnostics(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
