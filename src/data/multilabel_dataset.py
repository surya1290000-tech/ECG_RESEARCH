"""
Phase 6 Multi-Label Dataset Module for PTB-XL (5 Diagnostic Superclasses)
========================================================================

Parses all diagnostic statements per ECG record to create binary multi-hot vectors:
  y = [NORM, STTC, CD, MI, HYP] in {0, 1}^5

Maintains:
  - Exact Butterworth bandpass filtering (0.5 - 40 Hz, order 2)
  - Per-lead Z-score normalization
  - Exact patient-level GroupShuffleSplit (test_size=0.20, random_state=42)
  - 0 patient overlap across train (13,673), val (3,407), test (4,308)
"""

import ast
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt
from sklearn.model_selection import GroupShuffleSplit
import wfdb

from src.models.ecg_resnet import NUM_CLASSES, CLASS_NAMES, CLASS_TO_ID, ID_TO_CLASS
from configs.config import DATA_DIR, SAMPLING_RATE, SIGNAL_LENGTH, SPLIT_RANDOM_STATE


# =============================================================================
# Signal Preprocessing
# =============================================================================

def preprocess_ecg_signal(
    signal: np.ndarray,
    fs: int = 100,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 2,
) -> np.ndarray:
    """
    Standard preprocessing matching Phase 4/5:
      1. Butterworth bandpass filter (0.5 - 40 Hz)
      2. Transpose (1000, 12) -> (12, 1000)
      3. Per-lead Z-score normalization
    """
    signal = np.asarray(signal, dtype=np.float32)

    # Ensure shape (1000, 12)
    if signal.shape[0] != 1000 and signal.shape[1] == 12:
        if signal.shape[0] > 1000:
            signal = signal[:1000, :]
        else:
            padding = 1000 - signal.shape[0]
            signal = np.pad(signal, ((0, padding), (0, 0)), mode="edge")

    if signal.shape != (1000, 12):
        raise ValueError(f"Expected signal shape (1000, 12), got {signal.shape}")

    if not np.isfinite(signal).all():
        raise ValueError("ECG signal contains non-finite values (NaN or Inf)")

    # 1. Butterworth Bandpass Filter
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist

    if 0 < low < high < 1:
        b, a = butter(order, [low, high], btype="band")
        filtered_signal = np.zeros_like(signal)
        for i in range(12):
            filtered_signal[:, i] = filtfilt(b, a, signal[:, i])
    else:
        filtered_signal = signal

    # 2. Transpose (1000, 12) -> (12, 1000)
    filtered_signal = filtered_signal.T

    # 3. Per-lead Z-score normalization
    normalized_signal = np.zeros_like(filtered_signal)
    for lead_idx in range(12):
        lead = filtered_signal[lead_idx, :]
        mean = np.mean(lead)
        std = np.std(lead)
        if std > 1e-8:
            normalized_signal[lead_idx, :] = (lead - mean) / std
        else:
            normalized_signal[lead_idx, :] = lead - mean

    return normalized_signal.astype(np.float32)


# =============================================================================
# Metadata Loading & Multi-Hot Target Construction
# =============================================================================

def load_ptbxl_multilabel_metadata(
    data_dir: Path = DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load PTB-XL metadata and construct 5-dimensional multi-hot target vectors.
    """
    db_path = data_dir / "ptbxl_database.csv"
    scp_path = data_dir / "scp_statements.csv"

    if not db_path.exists():
        raise FileNotFoundError(f"ptbxl_database.csv not found at {db_path}")
    if not scp_path.exists():
        raise FileNotFoundError(f"scp_statements.csv not found at {scp_path}")

    df = pd.read_csv(db_path, index_col="ecg_id")
    scp_df = pd.read_csv(scp_path, index_col=0)

    # Parse scp_codes column
    def parse_scp(val):
        if pd.isna(val):
            return {}
        if isinstance(val, dict):
            return val
        try:
            return ast.literal_eval(val)
        except Exception:
            return {}

    df["scp_codes_parsed"] = df["scp_codes"].apply(parse_scp)

    # Extract diagnostic superclasses mapping
    diagnostic_map = scp_df[scp_df["diagnostic"] == 1]
    scp_to_superclass = {}
    for code, row in diagnostic_map.iterrows():
        superclass = row["diagnostic_class"]
        if pd.notna(superclass):
            scp_to_superclass[code] = superclass

    def get_multihot_vector(scp_dict) -> np.ndarray:
        multihot = np.zeros(NUM_CLASSES, dtype=np.float32)
        for code in scp_dict.keys():
            if code in scp_to_superclass:
                s_class = scp_to_superclass[code]
                for label in str(s_class).split(","):
                    label = label.strip()
                    if label in CLASS_TO_ID:
                        multihot[CLASS_TO_ID[label]] = 1.0
        return multihot

    df["multihot_targets"] = df["scp_codes_parsed"].apply(get_multihot_vector)
    df["label_count"] = df["multihot_targets"].apply(lambda v: int(np.sum(v)))

    # Filter out empty diagnostic labels (same as standard benchmark)
    df = df[df["label_count"] > 0].copy()

    return df, scp_df


def create_patient_level_multilabel_splits(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = SPLIT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Identical patient-level GroupShuffleSplit ensuring zero patient overlap.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss.split(df, groups=df["patient_id"]))

    train_val_df = df.iloc[train_val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df["patient_id"]))

    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()

    # Verify zero patient leakage
    train_patients = set(train_df["patient_id"])
    val_patients = set(val_df["patient_id"])
    test_patients = set(test_df["patient_id"])

    assert len(train_patients & val_patients) == 0, "Leakage between train and val!"
    assert len(train_patients & test_patients) == 0, "Leakage between train and test!"
    assert len(val_patients & test_patients) == 0, "Leakage between val and test!"

    return train_df, val_df, test_df


def create_ptbxl_fold10_splits(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Official PTB-XL benchmark split based on 'strat_fold' column:
      - Train: folds 1-8 (17,084 records)
      - Val: fold 9 (2,146 records)
      - Test: fold 10 (2,158 records)
    Guarantees zero patient overlap and preserves official benchmark protocol.
    """
    train_df = df[df["strat_fold"].between(1, 8)].copy()
    val_df = df[df["strat_fold"] == 9].copy()
    test_df = df[df["strat_fold"] == 10].copy()

    # Exact assertions as required
    assert train_df["strat_fold"].isin(range(1, 9)).all(), "Train split contains non-folds 1-8!"
    assert val_df["strat_fold"].eq(9).all(), "Validation split is not exactly fold 9!"
    assert test_df["strat_fold"].eq(10).all(), "Test split is not exactly fold 10!"

    # Verify zero patient leakage
    train_patients = set(train_df["patient_id"])
    val_patients = set(val_df["patient_id"])
    test_patients = set(test_df["patient_id"])

    assert len(train_patients & val_patients) == 0, "Leakage between train and val!"
    assert len(train_patients & test_patients) == 0, "Leakage between train and test!"
    assert len(val_patients & test_patients) == 0, "Leakage between val and test!"

    return train_df, val_df, test_df


# =============================================================================
# PyTorch Dataset (On-The-Fly)
# =============================================================================

class PTBXLMultiLabelECGDataset(Dataset):
    """
    PyTorch Dataset returning 12-lead ECG signals and 5-dimensional multi-hot targets.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        data_dir: Path = DATA_DIR,
        apply_preprocessing: bool = True,
    ):
        self.df = metadata_df.copy().reset_index(drop=False)
        self.data_dir = data_dir
        self.apply_preprocessing = apply_preprocessing

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        ecg_id = row["ecg_id"]
        rel_path = row["filename_lr"]

        record_path = self.data_dir / rel_path
        signal, _ = wfdb.rdsamp(str(record_path))

        if self.apply_preprocessing:
            signal = preprocess_ecg_signal(signal, fs=SAMPLING_RATE)
        else:
            signal = signal.T.astype(np.float32)

        multihot = row["multihot_targets"]

        return {
            "ecg": torch.tensor(signal, dtype=torch.float32),
            "labels": torch.tensor(multihot, dtype=torch.float32),
            "ecg_id": ecg_id,
        }


# =============================================================================
# High-Efficiency Bounded Array Loader (Single-Pass Preprocessing)
# =============================================================================

from concurrent.futures import ThreadPoolExecutor
import hashlib

def load_preprocessed_split_arrays(
    split_df: pd.DataFrame,
    data_dir: Path = DATA_DIR,
    cache_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads and applies standard Butterworth preprocessing with ThreadPool parallelization
    and transparent on-disk caching.
    Returns:
      X: np.ndarray of shape (N, 12, 1000), dtype float32
      Y: np.ndarray of shape (N, 5), dtype float32
      ecg_ids: np.ndarray of shape (N,), dtype int64
    """
    n = len(split_df)
    if cache_dir is None:
        cache_dir = data_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ecg_ids_arr = split_df.index.values if "ecg_id" not in split_df.columns else split_df["ecg_id"].values
    ecg_id_bytes = np.ascontiguousarray(ecg_ids_arr).tobytes()
    split_hash = hashlib.md5(ecg_id_bytes).hexdigest()[:12]
    cache_file = cache_dir / f"split_cache_n{n}_{split_hash}.npz"

    if cache_file.exists():
        if verbose:
            print(f"Loading {n} preprocessed records from cache: {cache_file.name}...")
        d = np.load(cache_file)
        return d["X"], d["Y"], d["ecg_ids"]

    t0 = time.time()
    if verbose:
        print(f"Loading and preprocessing {n} records (multi-threaded pass)...")

    df_reset = split_df.copy().reset_index(drop=False)

    def process_item(idx: int):
        row = df_reset.iloc[idx]
        ecg_id = row["ecg_id"]
        rel_path = row["filename_lr"]
        record_path = data_dir / rel_path
        signal, _ = wfdb.rdsamp(str(record_path))
        x = preprocess_ecg_signal(signal, fs=SAMPLING_RATE)
        y = row["multihot_targets"]
        return idx, x, y, ecg_id

    X = np.empty((n, 12, 1000), dtype=np.float32)
    Y = np.empty((n, NUM_CLASSES), dtype=np.float32)
    ecg_ids = np.empty(n, dtype=np.int64)

    with ThreadPoolExecutor(max_workers=8) as executor:
        for idx, x, y, ecg_id in executor.map(process_item, range(n)):
            X[idx] = x
            Y[idx] = y
            ecg_ids[idx] = ecg_id

    elapsed = time.time() - t0
    if verbose:
        mem_mb = X.nbytes / (1024 * 1024)
        print(f"Preprocessed {n} records in {elapsed:.2f}s ({mem_mb:.1f} MB in RAM).")

    try:
        np.savez(cache_file, X=X, Y=Y, ecg_ids=ecg_ids)
        if verbose:
            print(f"Saved preprocessed cache to {cache_file.name}")
    except Exception as e:
        if verbose:
            print(f"Could not save cache: {e}")

    return X, Y, ecg_ids


class TensorMultiLabelDataset(Dataset):
    """
    High-efficiency in-memory PyTorch Dataset wrapping pre-filtered tensors.
    Zero disk I/O and zero redundant SciPy filtering during batch iteration.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, ecg_ids: np.ndarray):
        assert len(X) == len(Y) == len(ecg_ids), "Length mismatch in TensorMultiLabelDataset"
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
        self.ecg_ids = ecg_ids

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "ecg": self.X[idx],
            "labels": self.Y[idx],
            "ecg_id": self.ecg_ids[idx],
        }
