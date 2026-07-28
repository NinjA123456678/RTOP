"""Create the response-confirmed controlled-anomaly experiment manifest."""

import argparse
from dataclasses import asdict
import json

from rtqp_phase1.experiments import RobustnessExperimentSpec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qos-type", choices=("rt", "tp"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(asdict(RobustnessExperimentSpec(args.qos_type, args.seed)), indent=2))

if __name__ == "__main__":
    main()
