"""
data_generator.py
─────────────────
Loads the real PitWall_Analytics_Cleaned.xlsx dataset and returns three
clean, typed DataFrames.

The file must live at:   data/PitWall_Analytics_Cleaned.xlsx
relative to this script (i.e. inside the repo root in a "data" folder).

If running on Streamlit Cloud and the local file is somehow missing,
it falls back to GITHUB_URL below.  Make sure to update GITHUB_URL with
your actual GitHub username and repo name before deploying.

Sheets used
  • Subscribers            800 rows × 18 cols
  • Engagement Sessions  29,240 rows × 10 cols
  • Revenue MRR             72 rows ×  9 cols
"""

from __future__ import annotations
import io
from pathlib import Path
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).parent

# The Excel file must be committed inside a "data/" folder in your repo.
LOCAL_XLSX = _HERE / "data" / "PitWall_Analytics_Cleaned.xlsx"

# !! Update YOUR_USERNAME and YOUR_REPO before deploying to Streamlit Cloud !!
# Example: "https://raw.githubusercontent.com/nivedhitha/F1---PitWall/main/data/PitWall_Analytics_Cleaned.xlsx"
GITHUB_URL = (
    "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main"
    "/data/PitWall_Analytics_Cleaned.xlsx"
)

_HEADER_ROW = 2     # The Excel file has real headers on row 3 (0-indexed = 2)


def _open_excel() -> dict[str, pd.DataFrame]:
    """Open the workbook — local first, GitHub fallback."""
    if LOCAL_XLSX.exists():
        return pd.read_excel(LOCAL_XLSX, sheet_name=None, header=_HEADER_ROW)
    # Fallback: fetch from GitHub (only used on Streamlit Cloud if local file is missing)
    import urllib.request
    try:
        with urllib.request.urlopen(GITHUB_URL, timeout=30) as resp:
            return pd.read_excel(
                io.BytesIO(resp.read()), sheet_name=None, header=_HEADER_ROW
            )
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not find the data file locally at '{LOCAL_XLSX}' "
            f"and the GitHub fallback also failed.\n"
            f"Make sure 'data/PitWall_Analytics_Cleaned.xlsx' is committed to your repo.\n"
            f"Original error: {exc}"
        ) from exc


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    subs  : subscriber-level DataFrame   (800 rows)
    sess  : session-level DataFrame      (29 240 rows)
    mrr   : monthly MRR summary          (72 rows)
    """
    xl   = _open_excel()
    subs = _clean_subscribers(xl["Subscribers"].copy())
    sess = _clean_sessions(xl["Engagement Sessions"].copy())
    mrr  = _clean_mrr(xl["Revenue MRR"].copy())
    return subs, sess, mrr


# ── Private cleaners ──────────────────────────────────────────────────────────

def _clean_subscribers(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    df["Signup Date"] = pd.to_datetime(df["Signup Date"], errors="coerce")
    df["Churn Date"]  = pd.to_datetime(df["Churn Date"],  errors="coerce")
    df["Churn Reason"] = df["Churn Reason"].fillna("Not Churned")
    df["churn_flag"]  = (df["Churned"] == "Yes").astype(int)
    return df


def _clean_sessions(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    df["Session Date"] = pd.to_datetime(df["Session Date"], errors="coerce")
    df["Is Weekend"]   = df["Is Weekend"].astype(bool)
    df["Engagement Score"]     = pd.to_numeric(df["Engagement Score"],     errors="coerce")
    df["Session Duration Min"] = pd.to_numeric(df["Session Duration Min"], errors="coerce")
    return df


def _clean_mrr(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m", errors="coerce")
    return df
