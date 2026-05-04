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
    Returns one concatenated dataframe with metadata and the actual hidden dimensions.
    """
    pairs = pair_meta_vec_files(run_dir)
    if not pairs:
        print(f"[INFO] No valid meta/vec pairs found in {run_dir}")
        return None

    dfs = []
    hidden_dim_seen = None

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

        if hidden_dim_seen is None:
            hidden_dim_seen = vec.shape[1]
        elif vec.shape[1] != hidden_dim_seen:
            raise ValueError(
                f"Hidden dimension mismatch in {run_dir}\n"
                f"  Previous hidden dim: {hidden_dim_seen}\n"
                f"  {vec_path.name}: {vec.shape}"
            )

        print(
            f"[LOAD] {vec_path.name}: shape={vec.shape}, "
            f"mean={vec.mean():.6f}, std={vec.std():.6f}, "
            f"min={vec.min():.6f}, max={vec.max():.6f}"
        )

        this_df = meta_df.copy()
        this_df["epoch"] = epoch
        this_df["condition"] = condition_name
        this_df["run"] = run_name

        for d in range(vec.shape[1]):
            this_df[f"dim_{d+1}"] = vec[:, d]

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

    dim_cols = sorted(
        [c for c in plot_df.columns if re.match(r"^dim_\d+$", c)],
        key=lambda x: int(x.split("_")[1])
    )

    if not dim_cols:
        raise ValueError("No hidden dimension columns found.")

    long_df = plot_df.melt(
        id_vars=["epoch", "condition", "run", "phoneme_plot"],
        value_vars=dim_cols,
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
    Make one subplot per actual hidden dimension.
    Plot:
      x = epoch
      y = mean dimension value
      color = phoneme
      shaded region = CI
    """
    summary = summarize_long_df(df_all, ci_level=ci_level)

    dims = sorted(
        summary["dimension"].unique(),
        key=lambda x: int(x.split("_")[1])
    )
    phonemes = sorted(summary["phoneme_plot"].unique())

    n_dims = len(dims)
    n_cols = 2 if n_dims > 1 else 1
    n_rows = int(np.ceil(n_dims / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=dims,
        shared_xaxes=True
    )

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

    for i, dim in enumerate(dims):
        r = i // n_cols + 1
        c = i % n_cols + 1

        dim_df = summary[summary["dimension"] == dim].copy()

        for ph in phonemes:
            sub = dim_df[dim_df["phoneme_plot"] == ph].sort_values("epoch")
            if sub.empty:
                continue

            line_color = phoneme_to_line[ph]
            fill_color = phoneme_to_fill[ph]

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

            fig.add_trace(
                go.Scatter(
                    x=sub["epoch"],
                    y=sub["mean"],
                    mode="lines",
                    line=dict(width=2, color=line_color),
                    name=ph,
                    legendgroup=ph,
                    showlegend=(dim == dims[0]),
                    customdata=np.stack(
                        [sub["n"], sub["ci_low"], sub["ci_high"]],
                        axis=-1
                    ),
                    hovertemplate=(
                        "Phoneme=%{fullData.name}<br>"
                        "Epoch=%{x}<br>"
                        "Mean=%{y:.6f}<br>"
                        "n=%{customdata[0]}<br>"
                        "CI=[%{customdata[1]:.6f}, %{customdata[2]:.6f}]"
                        "<extra></extra>"
                    )
                ),
                row=r, col=c
            )

    title = f"{df_all['condition'].iloc[0]} | run {df_all['run'].iloc[0]} | {len(dims)} hidden dims"

    fig.update_layout(
        title=title,
        height=450 * n_rows,
        width=1200,
        template="plotly_white",
        legend_title_text="Phoneme"
    )

    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Mean dimension value")

    fig.write_html(str(out_html))
    print(f"[SAVED] {out_html}")

def load_run_data_pca(run_dir: Path, condition_name: str, run_name: str):
    """
    Load all paired meta/vec files for one run, concatenate them,
    and return:
      df_all: metadata + vectors + epoch/condition/run
      X_all:  2D numpy array of raw encodings
    """
    pairs = pair_meta_vec_files(run_dir)
    if not pairs:
        print(f"[INFO] No valid meta/vec pairs found in {run_dir}")
        return None, None

    dfs = []
    all_vecs = []

    for item in pairs:
        epoch = item["epoch"]
        meta_path = item["meta_path"]
        vec_path = item["vec_path"]

        meta_df = pd.read_csv(meta_path)
        vec = np.load(vec_path)

        # Flatten if vectors are not already 2D
        if vec.ndim == 1:
            vec = vec[:, None]
        elif vec.ndim > 2:
            vec = vec.reshape(vec.shape[0], -1)

        # do PCA here
        pca = PCA(n_components=3)
        vec_pca = pca.fit_transform(vec)

        if len(meta_df) != vec.shape[0]:
            raise ValueError(
                f"Row mismatch in {run_dir}\n"
                f"  {meta_path.name}: {len(meta_df)} rows\n"
                f"  {vec_path.name}: {vec.shape[0]} rows"
            )

        this_df = meta_df.copy()
        this_df["epoch"] = epoch
        this_df["condition"] = condition_name
        this_df["run"] = run_name

        dfs.append(this_df)
        all_vecs.append(vec_pca)

    df_all = pd.concat(dfs, ignore_index=True)
    X_all = np.concatenate(all_vecs, axis=0)

    return df_all, X_all


def make_animated_plot(df_all: pd.DataFrame, X_all: np.ndarray, out_html: Path):
    # Determine phoneme column
    phoneme_col = PHONEME_COLUMN if PHONEME_COLUMN is not None else infer_phoneme_column(df_all)

    # # Fit PCA once across all epochs in this run
    # pca = PCA(n_components=3)
    # X_pca = pca.fit_transform(X_all)
    X_pca = X_all

    plot_df = df_all.copy()
    plot_df["PC1"] = X_pca[:, 0]
    plot_df["PC2"] = X_pca[:, 1]
    plot_df["PC3"] = X_pca[:, 2]
    plot_df["phoneme_plot"] = plot_df[phoneme_col].astype(str)

    # Sort frames
    plot_df = plot_df.sort_values(["epoch"]).reset_index(drop=True)

    title = (
        f"{plot_df['condition'].iloc[0]} | run {plot_df['run'].iloc[0]}"
        f"<br>PCA explained variance: "
        # f"{pca.explained_variance_ratio_[0]:.3f}, "
        # f"{pca.explained_variance_ratio_[1]:.3f}, "
        # f"{pca.explained_variance_ratio_[2]:.3f}"
    )

    fig = px.scatter_3d(
        plot_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="phoneme_plot",
        animation_frame="epoch",
        hover_data=["phoneme_plot"],
        title=title
    )

    fig.update_traces(marker=dict(size=1))

    # Keep axis ranges fixed across animation frames
    x_margin = (plot_df["PC1"].max() - plot_df["PC1"].min()) * 0.05 + 1e-9
    y_margin = (plot_df["PC2"].max() - plot_df["PC2"].min()) * 0.05 + 1e-9
    z_margin = (plot_df["PC3"].max() - plot_df["PC3"].min()) * 0.05 + 1e-9

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[plot_df["PC1"].min() - x_margin, plot_df["PC1"].max() + x_margin]),
            yaxis=dict(range=[plot_df["PC2"].min() - y_margin, plot_df["PC2"].max() + y_margin]),
            zaxis=dict(range=[plot_df["PC3"].min() - z_margin, plot_df["PC3"].max() + z_margin]),
        ),
        legend_title_text="Phoneme"
    )

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
    OUTPUT_DIR = VISUALIZATION_DIR / "plots_hidden_dims_mean_ci"
    VEC_OUTPUT_DIR = VISUALIZATION_DIR / "plots_pca_3d"
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    VEC_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
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

                out_html = OUTPUT_DIR / f"{condition_name}__run_{run_name}__mean_ci.html"
                make_2d_dimension_subplot_mean_ci(df_all, out_html, ci_level=0.95)

            except Exception as e:
                print(f"[ERROR] Failed on {run_dir}: {e}")

            try: 
                df_all, X_all = load_run_data_pca(run_dir, condition_name, run_name)
                if df_all is None:
                    continue
                out_html = VEC_OUTPUT_DIR / f"{condition_name}__run_{run_name}__pca3d_animation.html"
                make_animated_plot(df_all, X_all, out_html)
            except Exception as e: 
                print(f"[ERROR] Failed PCA plot on {run_dir}: {e}")


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Training script with config path')
    parser.add_argument('--config', type=str, required=True, help='Path to config.py')
    args = parser.parse_args()

    main(args.config)