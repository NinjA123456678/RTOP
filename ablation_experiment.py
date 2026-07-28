"""Print one or all Phase 1 ablation feature-flag configurations."""

import argparse
from dataclasses import asdict
import json

from rtqp_phase1.config import ablation_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(1, 9))
    args = parser.parse_args()
    cases = range(1, 9) if args.case is None else (args.case,)
    print(json.dumps([asdict(ablation_case(case)) for case in cases], indent=2))
    print("Phase 1 manifest only: unified model execution is intentionally deferred to Phase 2.")


if __name__ == "__main__":
    main()
