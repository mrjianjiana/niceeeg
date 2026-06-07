"""NICE with CLIP text prototype semantic regularization.

Baseline NICE aligns EEG features with pre-extracted CLIP image features.
This variant adds a second contrastive term that aligns EEG features with
CLIP text features generated from THINGS concept names.
"""

import argparse
import datetime
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ablation_config import CHANNEL_GROUPS, TIME_WINDOWS_MS
from nice_ablation import IE, torch


def parse_args():
    parser = argparse.ArgumentParser(description="NICE + CLIP text semantic loss")
    parser.add_argument("--dnn", default="clip", type=str)
    parser.add_argument("--epoch", default=200, type=int)
    parser.add_argument("--num_sub", default=10, type=int)
    parser.add_argument("--batch-size", default=250, type=int)
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--data_path", default="./Data/things-eeg2/Preprocessed_data_250Hz/")
    parser.add_argument("--feature_path", default="./dnn_feature/")
    parser.add_argument("--text_feature_path", default="./dnn_feature/text_clip_feature_maps_training.npy")
    parser.add_argument("--result_path", default="./text_semantic_runs/lambda_0.1/results/")
    parser.add_argument("--model_path", default="./text_semantic_runs/lambda_0.1/model/")
    parser.add_argument("--time_window", default="baseline",
                        help="Named window or explicit 'start:end' ms. "
                             f"Named: {', '.join(sorted(TIME_WINDOWS_MS))}")
    parser.add_argument("--channel_group", default="all",
                        choices=sorted(CHANNEL_GROUPS.keys()))
    parser.add_argument("--train_reps", default="all")
    parser.add_argument("--test_reps", default="all")
    parser.add_argument("--text_loss_weight", default=0.1, type=float)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


class TextSemanticIE(IE):
    def __init__(self, args, nsub):
        super().__init__(args, nsub)
        self.text_loss_weight = args.text_loss_weight
        self.log_write.write(
            f"Text semantic loss: weight={args.text_loss_weight}, "
            f"path={args.text_feature_path}\n"
        )
        print(f"Text semantic loss enabled: weight={args.text_loss_weight}")

    def model_prefix(self):
        base = super().model_prefix()
        return f"textw{self.text_loss_weight}_{base}"

    def get_aux_train_features(self):
        text_features = np.load(self.args.text_feature_path, allow_pickle=True)
        text_features = np.squeeze(text_features).astype(np.float32)
        return text_features

    def compute_aux_loss(self, eeg_features, aux_features, labels, logit_scale):
        if aux_features is None or self.text_loss_weight <= 0:
            return None
        text_features = aux_features / aux_features.norm(dim=1, keepdim=True)
        logits_per_eeg = logit_scale * eeg_features @ text_features.t()
        logits_per_text = logits_per_eeg.t()
        loss_eeg = self.criterion_cls(logits_per_eeg, labels)
        loss_text = self.criterion_cls(logits_per_text, labels)
        return self.text_loss_weight * (loss_eeg + loss_text) / 2


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
        ie = TextSemanticIE(args, i + 1)
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

