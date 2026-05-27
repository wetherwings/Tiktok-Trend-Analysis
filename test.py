#  Run: pytest tests/ -v

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder


# ── Helper ────────────────────────────────────────────────────

def make_sample_df(n=200):
    """Synthetic DataFrame matching transformer.ipynb columns."""
    np.random.seed(42)
    return pd.DataFrame({
        "video_view_count":         np.random.randint(100, 5_000_000, n),
        "video_duration_sec":       np.random.randint(5, 60, n),
        "video_transcription_text": [f"sample tiktok caption {i}" for i in range(n)],
        "verified_status":          np.random.choice(["verified", "not verified"], n),
        "author_ban_status":        np.random.choice(["active", "under review", "banned"], n),
    })


# ── Test 1: Virality label (transformer.ipynb Cell 10) ────────

def test_viral_label_is_binary():
    """is_viral must only contain 0 and 1."""
    df = make_sample_df()
    threshold = df["video_view_count"].median()
    df["is_viral"] = (df["video_view_count"] > threshold).astype(int)
    assert set(df["is_viral"].unique()).issubset({0, 1})


def test_viral_label_balanced():
    """Median split should produce ~50/50 class distribution."""
    df = make_sample_df()
    threshold = df["video_view_count"].median()
    df["is_viral"] = (df["video_view_count"] > threshold).astype(int)
    assert 0.3 <= df["is_viral"].mean() <= 0.7


def test_above_median_is_viral():
    """Every video above median view count must have label = 1."""
    df = make_sample_df()
    threshold = df["video_view_count"].median()
    df["is_viral"] = (df["video_view_count"] > threshold).astype(int)
    assert (df.loc[df["video_view_count"] > threshold, "is_viral"] == 1).all()


# ── Test 2: Numeric feature cleaning (transformer.ipynb Cell 10)

def test_duration_standardization():
    """
    Matches transformer.ipynb Cell 9:
    df[col] = (df[col] - mean) / (std + 1e-8)
    Result must be finite with no nulls.
    """
    df = make_sample_df()
    df["video_duration_sec"] = pd.to_numeric(
        df["video_duration_sec"], errors="coerce"
    ).fillna(0)
    mean = df["video_duration_sec"].mean()
    std  = df["video_duration_sec"].std()
    df["video_duration_sec"] = (df["video_duration_sec"] - mean) / (std + 1e-8)
    assert np.isfinite(df["video_duration_sec"].values).all()
    assert df["video_duration_sec"].isnull().sum() == 0


def test_label_encoder_produces_integers():
    """
    Matches transformer.ipynb Cell 10:
    LabelEncoder on verified_status and author_ban_status.
    """
    df = make_sample_df()
    le = LabelEncoder()
    for col in ["verified_status", "author_ban_status"]:
        df[col] = le.fit_transform(df[col].astype(str))
    assert df["verified_status"].isnull().sum() == 0
    assert df["author_ban_status"].isnull().sum() == 0
    assert df["verified_status"].nunique() <= 2
    assert df["author_ban_status"].nunique() <= 3


# ── Test 3: Text cleaning for DistilBERT ─────────────────────

def test_missing_transcription_filled():
    """
    Matches transformer.ipynb Cell 8:
    df['video_transcription_text'].fillna('').astype(str)
    Null values must become empty string, not NaN.
    """
    df = make_sample_df()
    df.loc[[0, 5, 10], "video_transcription_text"] = None
    df["video_transcription_text"] = (
        df["video_transcription_text"].fillna("").astype(str)
    )
    assert df["video_transcription_text"].isnull().sum() == 0
    assert df["video_transcription_text"].iloc[0] == ""


def test_tokenizer_input_is_valid_string():
    """
    DistilBERT tokenizer requires all inputs to be non-null strings.
    Matches transformer.ipynb Dataset __getitem__:
    text = str(self.df.loc[idx, 'video_transcription_text'])
    """
    df = make_sample_df()
    df["video_transcription_text"] = df["video_transcription_text"].astype(object)
    df.loc[0, "video_transcription_text"] = None
    df.loc[1, "video_transcription_text"] = 999
    df["video_transcription_text"] = (
        df["video_transcription_text"].fillna("").astype(str)
    )
    assert df["video_transcription_text"].apply(type).eq(str).all()
    assert df["video_transcription_text"].isnull().sum() == 0
