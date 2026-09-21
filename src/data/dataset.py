"""
PTB-XL Dataset Loading & Preprocessing Module.
==============================================

Handles:
  1. Preprocessing raw 12-lead ECG signals:
     - Bandpass filter (Butterworth 0.5–40 Hz at fs=100 Hz)
     - Transpose to (12, 1000)
     - Per-lead Z-score normalization
  2. Parsing PTB-XL diagnostic superclasses (NORM, MI, STTC, CD, HYP)
  3. Patient-level train / val / test splitting using GroupShuffleSplit (random_state=42)
  4. Zero patient overlap verification
  5. PyTorch Dataset implementation for 12-lead ECGs

Canonical Class Mapping:
  NORM: 0, MI: 1, STTC: 2, CD: 3, HYP: 4
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit
import wfdb

from src.models.ecg_resnet import CLASS_NAMES, CLASS_TO_ID, NUM_CLASSES
from configs.config import DATA_DIR, METADATA_CSV, SCP_STATEMENTS_CSV, RECORDS_DIR


# ---------------------------------------------------------------------------
# Signal Preprocessing
# ---------------------------------------------------------------------------

def preprocess_ecg(
    signal: np.ndarray,
    fs: float = 100.0,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """
    Preprocess one 12-lead ECG signal array.

    Input:
        signal : numpy array of shape (1000, 12) or (N, 12)

    Output:
        processed : float32 numpy array of shape (12, 1000)
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

    # 1. Butterworth Bandpass Filter (0.5 - 40 Hz)
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist

    if 0 < low < high < 1:
        b, a = butter(order, [low, high], btype="band")
        # Filter each lead along time axis
        filtered_signal = np.zeros_like(signal)
        for i in range(12):
            filtered_signal[:, i] = filtfilt(b, a, signal[:, i])
    else:
        filtered_signal = signal

    # 2. Transpose (1000, 12) -> (12, 1000)
    filtered_signal = filtered_signal.T  # (12, 1000)

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


# ---------------------------------------------------------------------------
# Label Parsing
# ---------------------------------------------------------------------------

def load_ptbxl_metadata(
    data_dir: Path = DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load ptbxl_database.csv and scp_statements.csv from data_dir."""
    meta_path = data_dir / "ptbxl_database.csv"
    scp_path = data_dir / "scp_statements.csv"

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file missing: {meta_path}")
    if not scp_path.exists():
        raise FileNotFoundError(f"SCP statements file missing: {scp_path}")

    df = pd.read_csv(meta_path, index_col="ecg_id")
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

    # Extract diagnostic superclasses
    diagnostic_map = scp_df[scp_df["diagnostic"] == 1]
    scp_to_superclass = {}
    for code, row in diagnostic_map.iterrows():
        superclass = row["diagnostic_class"]
        if pd.notna(superclass):
            scp_to_superclass[code] = superclass

    def get_superclasses(scp_dict):
        classes = set()
        for code in scp_dict.keys():
            if code in scp_to_superclass:
                s_class = scp_to_superclass[code]
                for label in str(s_class).split(","):
                    label = label.strip()
                    if label in CLASS_NAMES:
                        classes.add(label)
        return sorted(list(classes))

    df["diagnostic_classes"] = df["scp_codes_parsed"].apply(get_superclasses)

    # Filter out empty diagnostic labels
    df = df[df["diagnostic_classes"].apply(lambda x: isinstance(x, (list, tuple, set)) and len(x) > 0)].copy()

    # Match exact arth.ipynb collate_fn behavior: label_name = row["diagnostic_classes"][0]
    def get_primary_label(classes):
        if not classes or not isinstance(classes, (list, tuple)):
            return None
        label_name = classes[0]
        return CLASS_TO_ID.get(label_name, None)

    df["primary_label_id"] = df["diagnostic_classes"].apply(get_primary_label)
    df = df[df["primary_label_id"].notna()].copy()
    df["primary_label_id"] = df["primary_label_id"].astype(int)

    return df, scp_df


# ---------------------------------------------------------------------------
# Patient-Level Splitting
# ---------------------------------------------------------------------------

def create_patient_level_splits(
    df: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform GroupShuffleSplit on patient_id:
      - Split 1: 80% train_val, 20% test
      - Split 2: 80% train, 20% val (of train_val) -> 64% train, 16% val, 20% test
      - Guarantees ZERO patient overlap between splits.
    """
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df["patient_id"]))

    train_val_df = df.iloc[train_val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df["patient_id"]))

    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()

    # Verify zero patient overlap
    train_p = set(train_df["patient_id"])
    val_p = set(val_df["patient_id"])
    test_p = set(test_df["patient_id"])

    overlap_train_val = train_p & val_p
    overlap_train_test = train_p & test_p
    overlap_val_test = val_p & test_p

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise RuntimeError("Patient leakage detected across splits!")

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class PTBXLECGDataset(Dataset):
    """
    PyTorch Dataset for PTB-XL 12-lead ECGs.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        data_dir: Path = DATA_DIR,
        apply_preprocessing: bool = True,
    ):
        self.df = dataframe.reset_index(drop=False)  # ecg_id stored in column or index
        self.data_dir = Path(data_dir)
        self.apply_preprocessing = apply_preprocessing

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        rel_path = row["filename_lr"]

        # Full waveform file path without extension
        record_path = self.data_dir / rel_path

        # Load raw signal via WFDB
        signal, fields = wfdb.rdsamp(str(record_path))
        signal = signal.astype(np.float32)

        if self.apply_preprocessing:
            processed_signal = preprocess_ecg(signal, fs=fields["fs"])
        else:
            processed_signal = signal.T

        signal_tensor = torch.tensor(processed_signal, dtype=torch.float32)
        label_id = row["primary_label_id"]
        label_tensor = torch.tensor(label_id, dtype=torch.long)
        ecg_id = row["ecg_id"] if "ecg_id" in row else row.name

        return {
            "ecg": signal_tensor,
            "labels": label_tensor,
            "ecg_id": ecg_id,
        }
