"""
NICE EEG ablation runner.

This is a non-invasive copy of nice_stand.py with experiment switches for:
- time-window slicing on the preprocessed EEG tensor
- channel/brain-region selection
- repetition-count subsampling before averaging

The original nice_stand.py is not modified.
"""

import argparse
import datetime
import itertools
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
    import torch.nn.init as init
    from einops.layers.torch import Rearrange
    from torch import Tensor
    from torch.autograd import Variable
except ModuleNotFoundError:
    torch = None
    init = None
    Rearrange = None
    Variable = None
    Tensor = object

    class _TorchStub:
        Module = object
        Sequential = object

    nn = _TorchStub()

from ablation_config import CHANNEL_GROUPS, TIME_WINDOWS_MS
from ablation_config import resolve_channel_indices, resolve_time_window


gpus = [0]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))


def parse_args():
    parser = argparse.ArgumentParser(description="NICE EEG ablation experiments")
    parser.add_argument("--dnn", default="clip", type=str)
    parser.add_argument("--epoch", default=200, type=int)
    parser.add_argument("--num_sub", default=10, type=int)
    parser.add_argument("--batch-size", default=1000, type=int)
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--data_path", default="./Data/things-eeg2/Preprocessed_data_250Hz/")
    parser.add_argument("--feature_path", default="./dnn_feature/")
    parser.add_argument("--result_path", default="./results_ablation/")
    parser.add_argument("--model_path", default="./model_ablation/")
    parser.add_argument("--time_window", default="baseline",
                        help="Named window or explicit 'start:end' ms. "
                             f"Named: {', '.join(sorted(TIME_WINDOWS_MS))}")
    parser.add_argument("--channel_group", default="all",
                        choices=sorted(CHANNEL_GROUPS.keys()))
    parser.add_argument("--train_reps", default="all",
                        help="Training repetitions to average: all or an integer.")
    parser.add_argument("--test_reps", default="all",
                        help="Test repetitions to average: all or an integer.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Load one subject and print resulting shapes without training.")
    return parser.parse_args()


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def parse_rep_count(value):
    if str(value).lower() == "all":
        return None
    count = int(value)
    if count <= 0:
        raise ValueError("Repetition count must be positive or 'all'.")
    return count


def temporal_feature_count(n_times):
    # Conv2d temporal kernel 25 -> T - 24, then AvgPool2d kernel 51 stride 5.
    after_conv = n_times - 25 + 1
    if after_conv < 51:
        raise ValueError(
            f"Time window has {n_times} samples, too short for TSConv "
            "(needs at least 75 samples at 250 Hz)."
        )
    return (after_conv - 51) // 5 + 1


def load_state_dict_safely(path):
    """Load local checkpoints without noisy torch.load FutureWarning when supported."""
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)


class PatchEmbedding(nn.Module):
    def __init__(self, n_channels, emb_size=40):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (n_channels, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.tsconv(x)
        x = self.projection(x)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class EncEeg(nn.Sequential):
    def __init__(self, n_channels, emb_size=40):
        super().__init__(
            PatchEmbedding(n_channels=n_channels, emb_size=emb_size),
            FlattenHead(),
        )


class ProjEeg(nn.Sequential):
    def __init__(self, embedding_dim, proj_dim=768, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class ProjImg(nn.Sequential):
    def __init__(self, embedding_dim=768, proj_dim=768, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x):
        return x


class IE:
    def __init__(self, args, nsub):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.batch_size_test = 400
        self.n_epochs = args.epoch
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub

        self.eeg_data_path = args.data_path
        self.img_data_path = args.feature_path
        self.test_center_path = args.feature_path
        self.result_path = Path(args.result_path)
        self.model_path = Path(args.model_path)
        self.result_path.mkdir(parents=True, exist_ok=True)
        self.model_path.mkdir(parents=True, exist_ok=True)

        self.window_ms, self.window_idx = resolve_time_window(args.time_window)
        self.train_rep_count = parse_rep_count(args.train_reps)
        self.test_rep_count = parse_rep_count(args.test_reps)

        self.log_write = open(self.result_path / f"log_subject{self.nSub}.txt", "w")
        self.log_write.write(
            f"Config: time_window_ms={self.window_ms}, "
            f"channel_group={args.channel_group}, train_reps={args.train_reps}, "
            f"test_reps={args.test_reps}\n"
        )

        metadata = self.load_eeg_metadata()
        self.channel_indices, self.selected_channels = resolve_channel_indices(
            metadata["ch_names"], args.channel_group
        )
        n_channels = len(self.channel_indices)
        n_times = self.window_idx[1] - self.window_idx[0]
        embedding_dim = 40 * temporal_feature_count(n_times)

        print(
            f"Subject {self.nSub}: channels={n_channels} {self.selected_channels}, "
            f"time={self.window_ms}ms samples={self.window_idx}, "
            f"Proj_eeg embedding_dim={embedding_dim}"
        )
        self.log_write.write(
            f"Selected channels ({n_channels}): {self.selected_channels}\n"
            f"Time samples: {self.window_idx}, embedding_dim={embedding_dim}\n"
        )

        if args.dry_run:
            return
        if torch is None:
            raise RuntimeError("PyTorch is required for training. Use --dry_run for data checks only.")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.Enc_eeg = EncEeg(n_channels=n_channels).cuda()
        self.Proj_eeg = ProjEeg(embedding_dim=embedding_dim).cuda()
        self.Proj_img = ProjImg().cuda()
        self.Enc_eeg = nn.DataParallel(self.Enc_eeg, device_ids=[i for i in range(len(gpus))])
        self.Proj_eeg = nn.DataParallel(self.Proj_eeg, device_ids=[i for i in range(len(gpus))])
        self.Proj_img = nn.DataParallel(self.Proj_img, device_ids=[i for i in range(len(gpus))])
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def model_prefix(self):
        rep_tag = f"tr{self.args.train_reps}_te{self.args.test_reps}"
        return f"{self.args.time_window}_{self.args.channel_group}_{rep_tag}_sub{self.nSub}_"

    def subject_dir(self):
        return Path(self.eeg_data_path) / f"sub-{self.nSub:02d}"

    def load_eeg_metadata(self):
        path = self.subject_dir() / "preprocessed_eeg_training.npy"
        eeg_dict = np.load(path, allow_pickle=True)
        return {
            "ch_names": list(eeg_dict["ch_names"]),
            "times": np.asarray(eeg_dict["times"]),
        }

    def select_repetitions(self, data, count, rng):
        if count is None:
            return data
        if count > data.shape[1]:
            raise ValueError(
                f"Requested {count} repetitions, but data only has {data.shape[1]}."
            )
        indices = rng.choice(data.shape[1], size=count, replace=False)
        return data[:, np.sort(indices)]

    def prepare_eeg_tensor(self, eeg_dict, rep_count, rng):
        data = eeg_dict["preprocessed_eeg_data"]
        data = self.select_repetitions(data, rep_count, rng)
        data = data[:, :, self.channel_indices, self.window_idx[0]:self.window_idx[1]]
        data = np.mean(data, axis=1)
        data = np.expand_dims(data, axis=1)
        return data.astype(np.float32, copy=False)

    def get_eeg_data(self):
        rng = np.random.default_rng(self.args.seed + self.nSub)
        train_path = self.subject_dir() / "preprocessed_eeg_training.npy"
        test_path = self.subject_dir() / "preprocessed_eeg_test.npy"

        train_dict = np.load(train_path, allow_pickle=True)
        test_dict = np.load(test_path, allow_pickle=True)

        train_data = self.prepare_eeg_tensor(train_dict, self.train_rep_count, rng)
        test_data = self.prepare_eeg_tensor(test_dict, self.test_rep_count, rng)
        train_label = []
        test_label = np.arange(200)
        return train_data, train_label, test_data, test_label

    def get_image_data(self):
        train_img_feature = np.load(
            Path(self.img_data_path) / f"{self.args.dnn}_feature_maps_training.npy",
            allow_pickle=True,
        )
        test_img_feature = np.load(
            Path(self.img_data_path) / f"{self.args.dnn}_feature_maps_test.npy",
            allow_pickle=True,
        )
        return np.squeeze(train_img_feature), np.squeeze(test_img_feature)

    def train(self):
        if self.args.dry_run:
            train_eeg, _, test_eeg, _ = self.get_eeg_data()
            print(f"Dry run subject {self.nSub}: train_eeg={train_eeg.shape}, test_eeg={test_eeg.shape}")
            self.log_write.close()
            return 0.0, 0.0, 0.0

        self.Enc_eeg.apply(weights_init_normal)
        self.Proj_eeg.apply(weights_init_normal)
        self.Proj_img.apply(weights_init_normal)

        train_eeg, _, test_eeg, test_label = self.get_eeg_data()

        train_img_feature, _ = self.get_image_data()
        test_center = np.load(
            Path(self.test_center_path) / f"center_{self.args.dnn}.npy",
            allow_pickle=True,
        )

        train_shuffle = np.random.permutation(len(train_eeg))
        train_eeg = train_eeg[train_shuffle]
        train_img_feature = train_img_feature[train_shuffle]

        val_eeg = torch.from_numpy(train_eeg[:740])
        val_image = torch.from_numpy(train_img_feature[:740])
        train_eeg = torch.from_numpy(train_eeg[740:])
        train_image = torch.from_numpy(train_img_feature[740:])

        dataset = torch.utils.data.TensorDataset(train_eeg, train_image)
        self.dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=True
        )
        val_dataset = torch.utils.data.TensorDataset(val_eeg, val_image)
        self.val_dataloader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=self.batch_size, shuffle=False
        )

        test_eeg = torch.from_numpy(test_eeg)
        test_center = torch.from_numpy(test_center)
        test_label = torch.from_numpy(test_label)
        test_dataset = torch.utils.data.TensorDataset(test_eeg, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=self.batch_size_test, shuffle=False
        )

        self.optimizer = torch.optim.Adam(
            itertools.chain(
                self.Enc_eeg.parameters(),
                self.Proj_eeg.parameters(),
                self.Proj_img.parameters(),
            ),
            lr=self.lr,
            betas=(self.b1, self.b2),
        )

        best_loss_val = np.inf
        best_epoch = 0
        prefix = self.model_prefix()

        for e in range(self.n_epochs):
            self.Enc_eeg.train()
            self.Proj_eeg.train()
            self.Proj_img.train()

            for _, (eeg, img) in enumerate(self.dataloader):
                eeg = Variable(eeg.cuda().type(self.Tensor))
                img_features = Variable(img.cuda().type(self.Tensor))
                labels = torch.arange(eeg.shape[0])
                labels = Variable(labels.cuda().type(self.LongTensor))

                eeg_features = self.Proj_eeg(self.Enc_eeg(eeg))
                img_features = self.Proj_img(img_features)
                eeg_features = eeg_features / eeg_features.norm(dim=1, keepdim=True)
                img_features = img_features / img_features.norm(dim=1, keepdim=True)

                logit_scale = self.logit_scale.exp()
                logits_per_eeg = logit_scale * eeg_features @ img_features.t()
                logits_per_img = logits_per_eeg.t()
                loss_eeg = self.criterion_cls(logits_per_eeg, labels)
                loss_img = self.criterion_cls(logits_per_img, labels)
                loss = (loss_eeg + loss_img) / 2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.Enc_eeg.eval()
            self.Proj_eeg.eval()
            self.Proj_img.eval()
            with torch.no_grad():
                for _, (veeg, vimg) in enumerate(self.val_dataloader):
                    veeg = Variable(veeg.cuda().type(self.Tensor))
                    vimg_features = Variable(vimg.cuda().type(self.Tensor))
                    vlabels = torch.arange(veeg.shape[0])
                    vlabels = Variable(vlabels.cuda().type(self.LongTensor))

                    veeg_features = self.Proj_eeg(self.Enc_eeg(veeg))
                    vimg_features = self.Proj_img(vimg_features)
                    veeg_features = veeg_features / veeg_features.norm(dim=1, keepdim=True)
                    vimg_features = vimg_features / vimg_features.norm(dim=1, keepdim=True)

                    logit_scale = self.logit_scale.exp()
                    vlogits_per_eeg = logit_scale * veeg_features @ vimg_features.t()
                    vlogits_per_img = vlogits_per_eeg.t()
                    vloss_eeg = self.criterion_cls(vlogits_per_eeg, vlabels)
                    vloss_img = self.criterion_cls(vlogits_per_img, vlabels)
                    vloss = (vloss_eeg + vloss_img) / 2

                    if vloss <= best_loss_val:
                        best_loss_val = vloss
                        best_epoch = e + 1
                        torch.save(self.Enc_eeg.module.state_dict(), self.model_path / f"{prefix}Enc_eeg_cls.pth")
                        torch.save(self.Proj_eeg.module.state_dict(), self.model_path / f"{prefix}Proj_eeg_cls.pth")
                        torch.save(self.Proj_img.module.state_dict(), self.model_path / f"{prefix}Proj_img_cls.pth")

            print(
                "Epoch:", e,
                "  Cos eeg: %.4f" % loss_eeg.detach().cpu().numpy(),
                "  Cos img: %.4f" % loss_img.detach().cpu().numpy(),
                "  loss val: %.4f" % vloss.detach().cpu().numpy(),
            )
            self.log_write.write(
                "Epoch %d: Cos eeg: %.4f, Cos img: %.4f, loss val: %.4f\n"
                % (
                    e,
                    loss_eeg.detach().cpu().numpy(),
                    loss_img.detach().cpu().numpy(),
                    vloss.detach().cpu().numpy(),
                )
            )

        all_center = test_center
        total = 0
        top1 = 0
        top3 = 0
        top5 = 0

        self.Enc_eeg.load_state_dict(
            load_state_dict_safely(self.model_path / f"{prefix}Enc_eeg_cls.pth"),
            strict=False,
        )
        self.Proj_eeg.load_state_dict(
            load_state_dict_safely(self.model_path / f"{prefix}Proj_eeg_cls.pth"),
            strict=False,
        )
        self.Proj_img.load_state_dict(
            load_state_dict_safely(self.model_path / f"{prefix}Proj_img_cls.pth"),
            strict=False,
        )

        self.Enc_eeg.eval()
        self.Proj_eeg.eval()
        self.Proj_img.eval()

        with torch.no_grad():
            for _, (teeg, tlabel) in enumerate(self.test_dataloader):
                teeg = Variable(teeg.type(self.Tensor))
                tlabel = Variable(tlabel.type(self.LongTensor))
                all_center = Variable(all_center.type(self.Tensor))

                tfea = self.Proj_eeg(self.Enc_eeg(teeg))
                tfea = tfea / tfea.norm(dim=1, keepdim=True)
                similarity = (100.0 * tfea @ all_center.t()).softmax(dim=-1)
                _, indices = similarity.topk(5)

                tt_label = tlabel.view(-1, 1)
                total += tlabel.size(0)
                top1 += (tt_label == indices[:, :1]).sum().item()
                top3 += (tt_label == indices[:, :3]).sum().item()
                top5 += (tt_label == indices).sum().item()

        top1_acc = float(top1) / float(total)
        top3_acc = float(top3) / float(total)
        top5_acc = float(top5) / float(total)

        print("The test Top1-%.6f, Top3-%.6f, Top5-%.6f" % (top1_acc, top3_acc, top5_acc))
        self.log_write.write("The best epoch is: %d\n" % best_epoch)
        self.log_write.write(
            "The test Top1-%.6f, Top3-%.6f, Top5-%.6f\n"
            % (top1_acc, top3_acc, top5_acc)
        )
        self.log_write.close()
        return top1_acc, top3_acc, top5_acc


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
        ie = IE(args, i + 1)
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
