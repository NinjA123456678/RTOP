"""Create a multi-seed manifest; seed values must be supplied explicitly."""

import argparse
from dataclasses import asdict
import json

from rtqp_phase1.experiments import MultiSeedExperimentSpec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qos-type", choices=("rt", "tp"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    spec = MultiSeedExperimentSpec(args.qos_type, tuple(args.seeds))
    print(json.dumps(asdict(spec), indent=2))


if __name__ == "__main__":
    main()
