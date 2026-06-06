"""Shared settings for NICE EEG ablation experiments.

This file is intentionally separate from nice_stand.py so the original
reproduction script remains untouched.
"""

TIME_WINDOWS_MS = {
    "baseline": (0, 1000),
    "0_1000": (0, 1000),
    "0_800": (0, 800),
    "100_600": (100, 600),
    "100_500": (100, 500),
}


CHANNEL_GROUPS = {
    "all": None,
    # Posterior visual electrodes: occipital plus parieto-occipital sites.
    "occipital": ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"],
    "occipital_strict": ["O1", "Oz", "O2"],
    "temporal": ["FT7", "FT8", "T7", "T8", "TP7", "TP8", "TP9", "TP10"],
    "parietal": ["P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8"],
    "occipital_temporal": [
        "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
        "FT7", "FT8", "T7", "T8", "TP7", "TP8", "TP9", "TP10",
    ],
    "occipital_parietal": [
        "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
        "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    ],
}


def time_window_to_indices(window_ms, sfreq=250):
    """Convert a [start_ms, end_ms) window to sample indices."""
    start_ms, end_ms = window_ms
    start_idx = int(round(start_ms / 1000 * sfreq))
    end_idx = int(round(end_ms / 1000 * sfreq))
    if start_idx < 0 or end_idx <= start_idx:
        raise ValueError(f"Invalid time window: {window_ms}")
    return start_idx, end_idx


def resolve_time_window(name_or_range, sfreq=250):
    """Resolve a named window or an explicit 'start:end' ms string."""
    if name_or_range in TIME_WINDOWS_MS:
        window_ms = TIME_WINDOWS_MS[name_or_range]
    else:
        try:
            start_ms, end_ms = name_or_range.split(":", 1)
            window_ms = (int(start_ms), int(end_ms))
        except ValueError as exc:
            valid = ", ".join(sorted(TIME_WINDOWS_MS))
            raise ValueError(
                f"Unknown time window '{name_or_range}'. Use one of {valid} "
                "or an explicit 'start:end' ms range."
            ) from exc
    return window_ms, time_window_to_indices(window_ms, sfreq=sfreq)


def resolve_channel_indices(ch_names, group_name):
    """Resolve a channel group name to indices in the preprocessed EEG data."""
    if group_name not in CHANNEL_GROUPS:
        valid = ", ".join(sorted(CHANNEL_GROUPS))
        raise ValueError(f"Unknown channel group '{group_name}'. Use one of: {valid}")

    wanted = CHANNEL_GROUPS[group_name]
    if wanted is None:
        return list(range(len(ch_names))), list(ch_names)

    missing = [name for name in wanted if name not in ch_names]
    if missing:
        raise ValueError(f"Missing channels for group '{group_name}': {missing}")

    indices = [ch_names.index(name) for name in wanted]
    return indices, wanted

