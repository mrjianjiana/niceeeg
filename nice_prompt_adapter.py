"""NICE baseline with a lightweight EEG residual prompt adapter.

This script is separate from nice_stand.py. It reuses the ablation runner's
data slicing and result-directory logic, then adds an optional MLP adapter:

    prompt = MLP(eeg_feat)
    eeg_feat = eeg_feat + prompt

Normalization and contrastive loss stay unchanged.
"""

import argparse
import datetime
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ablation_config import CHANNEL_GROUPS, TIME_WINDOWS_MS
from nice_ablation import IE, load_state_dict_safely, nn, torch


def parse_args():
    parser = argparse.ArgumentParser(description="NICE with EEG-guided residual adapter")
    parser.add_argument("--dnn", default="clip", type=str)
    parser.add_argument("--epoch", default=200, type=int)
    parser.add_argument("--num_sub", default=10, type=int)
    parser.add_argument("--batch-size", default=250, type=int)
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--data_path", default="./Data/things-eeg2/Preprocessed_data_250Hz/")
    parser.add_argument("--feature_path", default="./dnn_feature/")
    parser.add_argument("--result_path", default="./prompt_adapter_runs/baseline_adapter/results/")
    parser.add_argument("--model_path", default="./prompt_adapter_runs/baseline_adapter/model/")
    parser.add_argument("--time_window", default="baseline",
                        help="Named window or explicit 'start:end' ms. "
                             f"Named: {', '.join(sorted(TIME_WINDOWS_MS))}")
    parser.add_argument("--channel_group", default="all",
                        choices=sorted(CHANNEL_GROUPS.keys()))
    parser.add_argument("--train_reps", default="all")
    parser.add_argument("--test_reps", default="all")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--use_prompt_adapter", action="store_true",
                        help="Enable the lightweight residual MLP adapter.")
    parser.add_argument("--prompt_hidden_dim", default=256, type=int)
    parser.add_argument("--prompt_type", default="residual", choices=["residual"])
    return parser.parse_args()


class PromptAdapter(nn.Module):
    def __init__(self, feat_dim=768, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feat_dim),
        )

    def forward(self, eeg_feat):
        return self.mlp(eeg_feat)


class PromptAdapterIE(IE):
    def __init__(self, args, nsub):
        super().__init__(args, nsub)
        self.use_prompt_adapter = args.use_prompt_adapter
        self.prompt_type = args.prompt_type

        if args.dry_run:
            return
        if torch is None:
            raise RuntimeError("PyTorch is required for prompt-adapter training.")

        if self.use_prompt_adapter:
            self.Prompt_adapter = PromptAdapter(
                feat_dim=768,
                hidden_dim=args.prompt_hidden_dim,
            ).cuda()
            self.log_write.write(
                f"PromptAdapter: type={args.prompt_type}, "
                f"hidden_dim={args.prompt_hidden_dim}\n"
            )
            print(
                f"PromptAdapter enabled: type={args.prompt_type}, "
                f"hidden_dim={args.prompt_hidden_dim}"
            )
        else:
            print("PromptAdapter disabled: running NICE baseline behavior.")

    def model_prefix(self):
        base = super().model_prefix()
        adapter = "adapter" if self.use_prompt_adapter else "no_adapter"
        hidden = getattr(self.args, "prompt_hidden_dim", "none")
        return f"{adapter}_h{hidden}_{base}"

    def adapt_eeg_features(self, eeg_features):
        if not self.use_prompt_adapter:
            return eeg_features
        prompt = self.Prompt_adapter(eeg_features)
        if self.prompt_type == "residual":
            return eeg_features + prompt
        raise ValueError(f"Unsupported prompt_type: {self.prompt_type}")

    def extra_trainable_parameters(self):
        if not self.use_prompt_adapter:
            return []
        return [self.Prompt_adapter.parameters()]

    def save_extra_modules(self, prefix):
        if self.use_prompt_adapter:
            torch.save(
                self.Prompt_adapter.state_dict(),
                self.model_path / f"{prefix}Prompt_adapter.pth",
            )

    def load_extra_modules(self, prefix):
        if self.use_prompt_adapter:
            self.Prompt_adapter.load_state_dict(
                load_state_dict_safely(self.model_path / f"{prefix}Prompt_adapter.pth"),
                strict=False,
            )


def main():
    args = parse_args()
    Path(args.result_path).mkdir(parents=True, exist_ok=True)
    Path(args.model_path).mkdir(parents=True, exist_ok=True)

    aver = []
    aver3 = []
    aver5 = []

    for i in range(args.num_sub):
        starttime = datetime.datetime.now()
        seed_n = args.seed + i
        print("seed is " + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        if torch is not None:
            torch.manual_seed(seed_n)
            torch.cuda.manual_seed(seed_n)
            torch.cuda.manual_seed_all(seed_n)

        print("Subject %d" % (i + 1))
        ie = PromptAdapterIE(args, i + 1)
        acc, acc3, acc5 = ie.train()
        print("THE BEST ACCURACY IS " + str(acc))
        print("subject %d duration: " % (i + 1) + str(datetime.datetime.now() - starttime))

        aver.append(acc)
        aver3.append(acc3)
        aver5.append(acc5)

        if args.dry_run:
            break

    aver.append(np.mean(aver))
    aver3.append(np.mean(aver3))
    aver5.append(np.mean(aver5))
    column = np.arange(1, len(aver)).tolist()
    column.append("ave")
    pd_all = pd.DataFrame(columns=column, data=[aver, aver3, aver5])
    pd_all.to_csv(Path(args.result_path) / "result.csv")


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))

