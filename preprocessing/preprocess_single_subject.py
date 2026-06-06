"""
Preprocess one subject from a directory shaped like:

Data/
  sub-01/
    ses-01/raw_eeg_training.npy
    ses-01/raw_eeg_test.npy
    ...

Default input in this project:
D:\\Ascienceproject\\NICE-EEG-main\\Data\\sub-01

Default output:
D:\\Ascienceproject\\NICE-EEG-main\\Data\\Preprocessed_data_250Hz\\sub-01
"""

import argparse
from pathlib import Path
import sys

from preprocessing_utils import mvnn
from preprocessing_utils import save_prepr


def parse_args():
	parser = argparse.ArgumentParser(
		description="Preprocess a single subject's EEG data."
	)
	default_subject_dir = Path(__file__).resolve().parents[1] / "Data" / "sub-01"
	parser.add_argument(
		"--subject_dir",
		default=str(default_subject_dir),
		type=str,
		help="Directory containing ses-XX/raw_eeg_*.npy for one subject.",
	)
	parser.add_argument("--sub", default=None, type=int, help="Subject id, e.g. 1 for sub-01.")
	parser.add_argument("--n_ses", default=4, type=int, help="Number of EEG sessions.")
	parser.add_argument("--sfreq", default=250, type=int, help="Downsampling frequency.")
	parser.add_argument(
		"--mvnn_dim",
		default="epochs",
		choices=["epochs", "time"],
		type=str,
		help="Compute MVNN covariance over epochs or time points.",
	)
	parser.add_argument(
		"--output_root",
		default=None,
		type=str,
		help="Directory that will contain Preprocessed_data_250Hz. Defaults to subject_dir's parent.",
	)
	return parser.parse_args()


def infer_subject_id(subject_dir):
	name = subject_dir.name
	if name.startswith("sub-"):
		return int(name.split("-", 1)[1])
	raise ValueError(
		"Could not infer subject id from directory name. "
		"Pass --sub explicitly, for example --sub 1."
	)


def validate_input(subject_dir, n_ses):
	missing = []
	for session in range(1, n_ses + 1):
		for data_part in ("training", "test"):
			path = subject_dir / f"ses-{session:02d}" / f"raw_eeg_{data_part}.npy"
			if not path.is_file():
				missing.append(str(path))
	if missing:
		raise FileNotFoundError("Missing raw EEG files:\n" + "\n".join(missing))


def epoching_single_subject(args, subject_dir, data_part, seed):
	"""Epoch one subject using raw files stored directly under subject_dir."""
	import mne
	import numpy as np
	from sklearn.utils import shuffle

	chan_order = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
				  'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
				  'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
				  'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
				  'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
				  'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
				  'O1', 'Oz', 'O2']

	epoched_data = []
	img_conditions = []
	for s in range(args.n_ses):
		eeg_path = subject_dir / f"ses-{s + 1:02d}" / f"raw_eeg_{data_part}.npy"
		eeg_data = np.load(eeg_path, allow_pickle=True).item()
		ch_names = eeg_data['ch_names']
		sfreq = eeg_data['sfreq']
		ch_types = eeg_data['ch_types']
		eeg_data = eeg_data['raw_eeg_data']

		info = mne.create_info(ch_names, sfreq, ch_types)
		raw = mne.io.RawArray(eeg_data, info)
		del eeg_data

		events = mne.find_events(raw, stim_channel='stim')
		raw.pick_channels(chan_order, ordered=True)
		idx_target = np.where(events[:, 2] == 99999)[0]
		events = np.delete(events, idx_target, 0)

		if args.sfreq < sfreq:
			raw, events = raw.resample(args.sfreq, events=events)
			epoch_tmax = 1.0 - 1.0 / args.sfreq
		else:
			epoch_tmax = 1.0

		epochs = mne.Epochs(raw, events, tmin=-.2, tmax=epoch_tmax, baseline=(None, 0),
			preload=True)
		del raw
		ch_names = epochs.info['ch_names']
		baseline_samples = int(round(0.2 * args.sfreq))
		times = epochs.times[baseline_samples:]

		data = epochs.get_data()
		events = epochs.events[:, 2]
		img_cond = np.unique(events)
		del epochs

		max_rep = 20 if data_part == 'test' else 2
		sorted_data = np.zeros((len(img_cond), max_rep, data.shape[1], data.shape[2]),
			dtype=np.float32)
		for i in range(len(img_cond)):
			idx = np.where(events == img_cond[i])[0]
			idx = shuffle(idx, random_state=seed, n_samples=max_rep)
			sorted_data[i] = data[idx]
		del data

		epoched_data.append(sorted_data[:, :, :, baseline_samples:])
		img_conditions.append(img_cond)
		del sorted_data

	return epoched_data, img_conditions, ch_names, times


def main():
	args = parse_args()
	subject_dir = Path(args.subject_dir).resolve()
	validate_input(subject_dir, args.n_ses)

	args.sub = args.sub if args.sub is not None else infer_subject_id(subject_dir)
	args.project_dir = str(Path(args.output_root).resolve() if args.output_root else subject_dir.parent)

	print(">>> Single-subject EEG preprocessing <<<")
	print("\nInput arguments:")
	for key, val in vars(args).items():
		print("{:16} {}".format(key, val))
	print("\nOutput directory:")
	print(Path(args.project_dir) / "Preprocessed_data_250Hz" / f"sub-{args.sub:02d}")

	seed = 20200220

	epoched_test, _, ch_names, times = epoching_single_subject(args, subject_dir, "test", seed)
	epoched_train, img_conditions_train, _, _ = epoching_single_subject(
		args, subject_dir, "training", seed
	)

	whitened_test, whitened_train = mvnn(args, epoched_test, epoched_train)
	del epoched_test, epoched_train

	save_prepr(
		args,
		whitened_test,
		whitened_train,
		img_conditions_train,
		ch_names,
		times,
		seed,
	)

	print("\nDone.")


if __name__ == "__main__":
	try:
		main()
	except Exception as exc:
		print(f"\nPreprocessing failed: {exc}", file=sys.stderr)
		raise
