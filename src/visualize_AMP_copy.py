import argparse
import importlib.util
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import plotly.offline as pyo
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os
import re
from pathlib import Path
import plotly.express as px

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================
# Configuration
# =========================

PHONEME_COLUMN = "vowel"

# Example:
# meta_E10_L2.csv
# vec_E10_L2.npy
META_PATTERN = re.compile(r"^meta_E(\d+)_.*\.csv$")
VEC_PATTERN  = re.compile(r"^vec_E(\d+)_.*\.npy$")


# =========================
# Helper functions
# =========================
def infer_phoneme_column(df: pd.DataFrame) -> str:
    return PHONEME_COLUMN


def get_epoch_from_meta_name(filename: str):
    m = META_PATTERN.match(filename)
    if m:
        return int(m.group(1))
    return None


def get_epoch_from_vec_name(filename: str):
    m = VEC_PATTERN.match(filename)
    if m:
        return int(m.group(1))
    return None


def pair_meta_vec_files(run_dir: Path):
    """
    Pair files like:
      meta_E10_L2.csv
      vec_E10_L2.npy
    by their shared suffix.
    """
    files = [p for p in run_dir.iterdir() if p.is_file()]

    meta_files = {}
    vec_files = {}

    for f in files:
        if f.suffix.lower() == ".csv" and f.name.startswith("meta_"):
            epoch = get_epoch_from_meta_name(f.name)
            if epoch is not None:
                key = f.name[len("meta_"):-4]   # remove "meta_" and ".csv"
                meta_files[key] = (epoch, f)

        elif f.suffix.lower() == ".npy" and f.name.startswith("vec_"):
            epoch = get_epoch_from_vec_name(f.name)
            if epoch is not None:
                key = f.name[len("vec_"):-4]    # remove "vec_" and ".npy"
                vec_files[key] = (epoch, f)

    paired = []
    shared_keys = sorted(set(meta_files.keys()) & set(vec_files.keys()))

    for key in shared_keys:
        meta_epoch, meta_path = meta_files[key]
        vec_epoch, vec_path = vec_files[key]

        if meta_epoch != vec_epoch:
            print(f"[WARN] Epoch mismatch in {run_dir}: {meta_path.name} vs {vec_path.name}")
            continue

        paired.append({
            "epoch": meta_epoch,
            "meta_path": meta_path,
            "vec_path": vec_path
        })

    paired.sort(key=lambda x: x["epoch"])
    return paired


def load_run_data(run_dir: Path, condition_name: str, run_name: str):
    """
    Load all paired meta/vec files for one run.
    Returns one concatenated dataframe with metadata and 4 dimensions.
    """
    pairs = pair_meta_vec_files(run_dir)
    if not pairs:
        print(f"[INFO] No valid meta/vec pairs found in {run_dir}")
        return None

    dfs = []

    for item in pairs:
        epoch = item["epoch"]
        meta_path = item["meta_path"]
        vec_path = item["vec_path"]

        meta_df = pd.read_csv(meta_path)
        vec = np.load(vec_path)

        if vec.ndim == 1:
            vec = vec[:, None]
        elif vec.ndim > 2:
            vec = vec.reshape(vec.shape[0], -1)

        if len(meta_df) != vec.shape[0]:
            raise ValueError(
                f"Row mismatch in {run_dir}\n"
                f"  {meta_path.name}: {len(meta_df)} rows\n"
                f"  {vec_path.name}: {vec.shape[0]} rows"
            )

        # if vec.shape[1] != 4:
        #     raise ValueError(
        #         f"Expected 4 dimensions in {vec_path.name}, but got shape {vec.shape}"
        #     )
        if vec.shape[1] < 4:
            # Pad with NaNs if less than 4 dimensions
            padded_vec = np.full((vec.shape[0], 4), 0)
            padded_vec[:, :vec.shape[1]] = vec
            vec = padded_vec

        this_df = meta_df.copy()
        this_df["epoch"] = epoch
        this_df["condition"] = condition_name
        this_df["run"] = run_name

        this_df["dim_1"] = vec[:, 0]
        this_df["dim_2"] = vec[:, 1]
        this_df["dim_3"] = vec[:, 2]
        this_df["dim_4"] = vec[:, 3]

        dfs.append(this_df)

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def summarize_long_df(df_all: pd.DataFrame, ci_level=0.95):
    """
    Convert to long format and compute mean + CI for each:
      epoch × phoneme × dimension
    """
    phoneme_col = infer_phoneme_column(df_all)

    plot_df = df_all.copy()
    plot_df["phoneme_plot"] = plot_df[phoneme_col].astype(str)

    long_df = plot_df.melt(
        id_vars=["epoch", "condition", "run", "phoneme_plot"],
        value_vars=["dim_1", "dim_2", "dim_3", "dim_4"],
        var_name="dimension",
        value_name="value"
    )

    summary = (
        long_df
        .groupby(["epoch", "phoneme_plot", "dimension"], as_index=False)
        .agg(
            mean=("value", "mean"),
            std=("value", "std"),
            n=("value", "size")
        )
    )

    summary["std"] = summary["std"].fillna(0.0)
    summary["sem"] = summary["std"] / np.sqrt(summary["n"])

    if ci_level == 0.95:
        z = 1.96
    elif ci_level == 0.99:
        z = 2.576
    elif ci_level == 0.90:
        z = 1.645
    else:
        z = 1.96

    summary["ci_low"] = summary["mean"] - z * summary["sem"]
    summary["ci_high"] = summary["mean"] + z * summary["sem"]

    return summary


def make_2d_dimension_subplot_mean_ci(df_all: pd.DataFrame, out_html: Path, ci_level=0.95):
    """
    Make 4 subplots:
      one per dimension
    Plot:
      x = epoch
      y = mean dimension value
      color = phoneme
      shaded region = CI
    """
    summary = summarize_long_df(df_all, ci_level=ci_level)

    dims = ["dim_1", "dim_2", "dim_3", "dim_4"]
    phonemes = sorted(summary["phoneme_plot"].unique())

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=dims,
        shared_xaxes=True
    )

    dim_to_pos = {
        "dim_1": (1, 1),
        "dim_2": (1, 2),
        "dim_3": (2, 1),
        "dim_4": (2, 2),
    }

    line_palette = [
        "rgb(99,110,250)",
        "rgb(239,85,59)",
        "rgb(0,204,150)",
        "rgb(171,99,250)",
        "rgb(255,161,90)",
        "rgb(25,211,243)",
        "rgb(255,102,146)",
        "rgb(182,232,128)",
        "rgb(255,151,255)",
        "rgb(254,203,82)"
    ]
    fill_palette = [
        "rgba(99,110,250,0.20)",
        "rgba(239,85,59,0.20)",
        "rgba(0,204,150,0.20)",
        "rgba(171,99,250,0.20)",
        "rgba(255,161,90,0.20)",
        "rgba(25,211,243,0.20)",
        "rgba(255,102,146,0.20)",
        "rgba(182,232,128,0.20)",
        "rgba(255,151,255,0.20)",
        "rgba(254,203,82,0.20)"
    ]

    phoneme_to_line = {
        ph: line_palette[i % len(line_palette)]
        for i, ph in enumerate(phonemes)
    }
    phoneme_to_fill = {
        ph: fill_palette[i % len(fill_palette)]
        for i, ph in enumerate(phonemes)
    }

    for dim in dims:
        r, c = dim_to_pos[dim]
        dim_df = summary[summary["dimension"] == dim].copy()

        for ph in phonemes:
            sub = dim_df[dim_df["phoneme_plot"] == ph].sort_values("epoch")
            if sub.empty:
                continue

            line_color = phoneme_to_line[ph]
            fill_color = phoneme_to_fill[ph]

            # Upper CI boundary
            fig.add_trace(
                go.Scatter(
                    x=sub["epoch"],
                    y=sub["ci_high"],
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=ph
                ),
                row=r, col=c
            )

            # Lower CI boundary + fill
            fig.add_trace(
                go.Scatter(
                    x=sub["epoch"],
                    y=sub["ci_low"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=fill_color,
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=ph
                ),
                row=r, col=c
            )

            # Mean line
            fig.add_trace(
                go.Scatter(
                    x=sub["epoch"],
                    y=sub["mean"],
                    mode="lines",
                    line=dict(width=2, color=line_color),
                    name=ph,
                    legendgroup=ph,
                    showlegend=(dim == "dim_1"),
                    customdata=np.stack(
                        [sub["n"], sub["ci_low"], sub["ci_high"]],
                        axis=-1
                    ),
                    hovertemplate=(
                        "Phoneme=%{fullData.name}<br>"
                        "Epoch=%{x}<br>"
                        "Mean=%{y:.4f}<br>"
                        "n=%{customdata[0]}<br>"
                        "CI=[%{customdata[1]:.4f}, %{customdata[2]:.4f}]"
                        "<extra></extra>"
                    )
                ),
                row=r, col=c
            )

    title = f"{df_all['condition'].iloc[0]} | run {df_all['run'].iloc[0]}"

    fig.update_layout(
        title=title,
        height=900,
        width=1200,
        template="plotly_white",
        legend_title_text="Phoneme"
    )

    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Mean dimension value")

    fig.write_html(str(out_html))
    print(f"[SAVED] {out_html}")

def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

# =========================
# Main
# =========================
def main(config_path):
    config = load_config(config_path)
    INCLUDE_PREFIX = config.INCLUDE_PREFIX if hasattr(config, "INCLUDE_PREFIX") else ""
    BASE_DIR = config.BASE_DIR if hasattr(config, "BASE_DIR") else Path(".")
    
    VECTOR_DIR = BASE_DIR / "eval_outputs"
    VISUALIZATION_DIR = BASE_DIR / "visualizations"
    OUTPUT_DIR = VISUALIZATION_DIR / "plots_2d_4dims_mean_ci"
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    if not VECTOR_DIR.exists():
        raise FileNotFoundError(f"Base directory not found: {VECTOR_DIR}")

    condition_dirs = [
        p for p in VECTOR_DIR.iterdir()
        if p.is_dir() and not p.name.endswith(".csv")
    ]

    condition_dirs = [
        p for p in condition_dirs
        if p.name.startswith(f"{INCLUDE_PREFIX}")
    ]

    for condition_dir in sorted(condition_dirs):
        condition_name = condition_dir.name

        run_dirs = [
            p for p in condition_dir.iterdir()
            if p.is_dir() and not p.name.endswith(".csv")
        ]

        for run_dir in sorted(run_dirs, key=lambda p: p.name):
            run_name = run_dir.name
            print(f"[PROCESSING] condition={condition_name}, run={run_name}")

            try:
                df_all = load_run_data(run_dir, condition_name, run_name)
                if df_all is None:
                    continue

                out_html = OUTPUT_DIR / f"{condition_name}__run_{run_name}__4dim_mean_ci.html"
                make_2d_dimension_subplot_mean_ci(df_all, out_html, ci_level=0.95)

            except Exception as e:
                print(f"[ERROR] Failed on {run_dir}: {e}")


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Training script with config path')
    parser.add_argument('--config', type=str, required=True, help='Path to config.py')
    args = parser.parse_args()

    main(args.config)