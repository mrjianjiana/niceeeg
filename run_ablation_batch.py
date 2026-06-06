"""Sequential runner for the most useful NICE ablation settings.

Examples:
    python run_ablation_batch.py --num_sub 1 --epoch 2
    python run_ablation_batch.py --num_sub 10 --epoch 200
"""

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_EXPERIMENTS = [
    ("time_0_800", {"time_window": "0_800", "channel_group": "all"}),
    ("time_100_600", {"time_window": "100_600", "channel_group": "all"}),
    ("time_100_500", {"time_window": "100_500", "channel_group": "all"}),
    ("channels_occipital", {"time_window": "baseline", "channel_group": "occipital"}),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run NICE ablations sequentially.")
    parser.add_argument("--num_sub", default=10, type=int)
    parser.add_argument("--epoch", default=200, type=int)
    parser.add_argument("--batch-size", default=1000, type=int)
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--base_result_dir", default="./ablation_runs")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Optional experiment names to run.")
    return parser.parse_args()


def main():
    args = parse_args()
    base = Path(args.base_result_dir)
    base.mkdir(parents=True, exist_ok=True)

    experiments = DEFAULT_EXPERIMENTS
    if args.only:
        wanted = set(args.only)
        experiments = [exp for exp in experiments if exp[0] in wanted]
        unknown = wanted - {name for name, _ in DEFAULT_EXPERIMENTS}
        if unknown:
            raise ValueError(f"Unknown experiments: {sorted(unknown)}")

    for name, opts in experiments:
        result_path = base / name / "results"
        model_path = base / name / "model"
        cmd = [
            sys.executable,
            "nice_ablation.py",
            "--num_sub", str(args.num_sub),
            "--epoch", str(args.epoch),
            "--batch-size", str(args.batch_size),
            "--seed", str(args.seed),
            "--time_window", opts["time_window"],
            "--channel_group", opts["channel_group"],
            "--result_path", str(result_path),
            "--model_path", str(model_path),
        ]
        print("\n>>> Running", name)
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

