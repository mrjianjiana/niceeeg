"""Sequential runner for repetition-count experiments.

Default design:
- train_reps stays "all"
- test_reps varies across 1, 5, 10, and all

This answers the application question: if the model is trained with all
available repetitions, how much does performance drop when test-time EEG is
averaged from fewer repetitions?
"""

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_TEST_REPS = ["1", "5", "10", "all"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run NICE repetition ablations.")
    parser.add_argument("--num_sub", default=10, type=int)
    parser.add_argument("--epoch", default=200, type=int)
    parser.add_argument("--batch-size", default=250, type=int)
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--base_result_dir", default="./repetition_runs")
    parser.add_argument("--train_reps", default="all")
    parser.add_argument("--test_reps", nargs="*", default=DEFAULT_TEST_REPS)
    parser.add_argument("--time_window", default="baseline")
    parser.add_argument("--channel_group", default="all")
    return parser.parse_args()


def main():
    args = parse_args()
    base = Path(args.base_result_dir)
    base.mkdir(parents=True, exist_ok=True)

    for test_reps in args.test_reps:
        name = f"train_reps_{args.train_reps}_test_reps_{test_reps}"
        result_path = base / name / "results"
        model_path = base / name / "model"
        cmd = [
            sys.executable,
            "nice_ablation.py",
            "--num_sub", str(args.num_sub),
            "--epoch", str(args.epoch),
            "--batch-size", str(args.batch_size),
            "--seed", str(args.seed),
            "--time_window", args.time_window,
            "--channel_group", args.channel_group,
            "--train_reps", str(args.train_reps),
            "--test_reps", str(test_reps),
            "--result_path", str(result_path),
            "--model_path", str(model_path),
        ]
        print("\n>>> Running", name)
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

