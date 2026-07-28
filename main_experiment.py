"""Create a Phase 1 main-experiment manifest without invoking legacy training."""

import argparse
from dataclasses import asdict
import json

from rtqp_phase1.experiments import MainExperimentSpec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qos-type", choices=("rt", "tp"), required=True)
    parser.add_argument("--density", type=float, required=True)
    args = parser.parse_args()
    spec = MainExperimentSpec(args.qos_type, args.density)
    print(json.dumps(asdict(spec), indent=2))
    print("Phase 1 manifest only: unified model execution is intentionally deferred to Phase 2.")


if __name__ == "__main__":
    main()
