import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import random
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances

import os

from sklearn.metrics import silhouette_score
from itertools import combinations

import importlib.util
import argparse

def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def select_pairs(indices_list, size):
    all_pairs = [(indices_list[i], indices_list[j]) for i in range(len(indices_list)) for j in range(i + 1, len(indices_list))]
    selected_pairs = random.sample(all_pairs, min(len(all_pairs), size))
    return selected_pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script with config path')
    parser.add_argument('--config', type=str, required=True, help='Path to config.py')
    args = parser.parse_args()

    config = load_config(args.config)

    run_names = config.RUN_NAMES
    run_names_short = config.RUN_NAMES_SHORT

    vector_storage_path = config.READ_BASE_PATH
    for run_name, run_name_short in zip(run_names, run_names_short):
        save_path = f"{config.WRITE_BASE_PATH}/observe_outputs/{run_name_short}/"
        os.makedirs(save_path, exist_ok=True)
        for run_time in range(config.RUN_TIMES_START, config.RUN_TIMES_END): 
            for epoch in range(config.EPOCH_START, config.EPOCH_END, config.EPOCH_INTERVAL):
                # -------------------------------
                # Load data
                # -------------------------------
                run_name_path = os.path.join(".", vector_storage_path, run_name, f"{run_time}")
                vec_L2 = np.load(os.path.join(run_name_path, f"vec_E{epoch}_L2.npy"))           # shape (N, 4)
                meta_L2 = pd.read_csv(os.path.join(run_name_path, f"meta_E{epoch}_L2.csv"))     # must align row-by-row
                assert len(vec_L2) == len(meta_L2), "Vector and metadata length mismatch (L2)!"
                vec_L1 = np.load(os.path.join(run_name_path, f"vec_E{epoch}_L1.npy")) # shape (N, 4)
                meta_L1 = pd.read_csv(os.path.join(run_name_path, f"meta_E{epoch}_L1.csv")) # must align row-by-row
                assert len(vec_L1) == len(meta_L1), "Vector and metadata length mismatch (L1)!"
        
                vecs = np.vstack((vec_L2, vec_L1))
                metas = pd.concat([meta_L2, meta_L1], axis=0, ignore_index=True)

                # -----------------------------------------------
                # NEW: vowel-by-vowel pairwise computation
                # -----------------------------------------------
                vowels = metas["vowel"].unique()
                sil_results = []

                consonant_list = metas["consonant"].unique().tolist()

                rng = np.random.default_rng(12345)  # stable randomness

                for vowel in vowels:

                    mask_vowel = metas["vowel"] == vowel
                    vec_v = vecs[mask_vowel]
                    meta_v = metas[mask_vowel].reset_index(drop=True)

                    # PRECOMPUTE distance matrix ONCE
                    # D_v = pairwise_distances(vec_v, metric="euclidean")

                    # restrict to consonants present in this vowel
                    consonants_here = meta_v["consonant"].unique().tolist()
                    consonants_here = [c for c in consonant_list if c in consonants_here]

                    C = len(consonants_here)
                    if C == 0:
                        continue

                    # ------------------------------------------------
                    # all consonant pairs including self-pair
                    # BUT only i ≤ j to avoid AB / BA duplicates
                    # ------------------------------------------------
                    for i in range(C):
                        for j in range(i, C):
                            c1 = consonants_here[i]
                            c2 = consonants_here[j]

                            if ((c1 == "ts") and (c2 == "c")) or ((c2 == "ts") and (c1 == "c")): 
                                if vowel in ["i", "e", "u", "o"]: 
                                    diff_trained = 'trained'
                                else: 
                                    diff_trained = 'vowel_untrained'
                            elif c1 == c2: 
                                if c1 in ["ts", "c"]: 
                                    if vowel in ["i", "e", "u", "o"]: 
                                        diff_trained = "trained"
                                    else: 
                                        diff_trained = "vowel_untrained"
                                else: 
                                    if vowel in ["i", "e", "u", "o"]: 
                                        diff_trained = 'consonant_untrained'
                                    else: 
                                        diff_trained = 'untrained'
                            else: 
                                if vowel in ["i", "e", "u", "o"]: 
                                    diff_trained = 'consonant_untrained'
                                else: 
                                    diff_trained = 'untrained'


                            # indices for this vowel
                            idx_c1 = meta_v.index[meta_v["consonant"] == c1].to_numpy()
                            idx_c2 = meta_v.index[meta_v["consonant"] == c2].to_numpy()

                            # -------------------------------
                            # Case 1: Same-category pair
                            # -------------------------------
                            if c1 == c2:
                                n = len(idx_c1)
                                if n < 4:
                                    # need at least 2 per artificial class
                                    sil_results.append({
                                        "vowel": vowel,
                                        "sil_type": f"{c1}_{c1}",
                                        "sil_score": np.nan,
                                        "training": diff_trained, 
                                        "cate_num": 2,
                                        "n_sample": 0
                                    })
                                    continue

                                # sample half for A, half for B
                                k = n // 2
                                idxA = rng.choice(idx_c1, size=k, replace=False)
                                remain = np.setdiff1d(idx_c1, idxA)
                                idxB = rng.choice(remain, size=k, replace=False)

                                vec_sub = np.vstack([vec_v[idxA], vec_v[idxB]])
                                # sub_idx = np.concatenate([idxA, idxB])
                                labels_sub = np.array([c1 + "_A"] * k + [c1 + "_B"] * k)

                                # D_sub = D_v[np.ix_(sub_idx, sub_idx)]
                                score = silhouette_score(vec_sub, labels_sub)
                                # score = silhouette_score(D_sub, labels_sub, metric="precomputed")

                                sil_results.append({
                                    "vowel": vowel,
                                    "sil_type": f"{c1}_{c1}",
                                    "sil_score": score,
                                    "training": diff_trained, 
                                    "cate_num": 2,
                                    "n_sample": 2 * k
                                })

                            # -------------------------------
                            # Case 2: Different-category pair
                            # -------------------------------
                            else:
                                n1 = len(idx_c1)
                                n2 = len(idx_c2)
                                if n1 < 2 or n2 < 2:
                                    sil_results.append({
                                        "vowel": vowel,
                                        "sil_type": f"{c1}_{c2}",
                                        "sil_score": np.nan,
                                        "training": diff_trained, 
                                        "cate_num": 2,
                                        "n_sample": 0
                                    })
                                    continue

                                k1 = n1 // 2
                                k2 = n2 // 2
                                k = min(k1, k2)     # ensure balanced

                                idx1 = rng.choice(idx_c1, size=k, replace=False)
                                idx2 = rng.choice(idx_c2, size=k, replace=False)

                                vec_sub = np.vstack([vec_v[idx1], vec_v[idx2]])
                                # sub_idx = np.concatenate([idx1, idx2])
                                labels_sub = np.array([c1] * k + [c2] * k)

                                if len(np.unique(labels_sub)) < 2:
                                    score = np.nan
                                else:
                                    # D_sub = D_v[np.ix_(sub_idx, sub_idx)]
                                    score = silhouette_score(vec_sub, labels_sub)
                                    # score = silhouette_score(D_sub, labels_sub, metric="precomputed")

                                sil_results.append({
                                    "vowel": vowel,
                                    "sil_type": f"{c1}_{c2}",
                                    "sil_score": score,
                                    "training": diff_trained, 
                                    "cate_num": 2,
                                    "n_sample": 2 * k
                                })

                # Save
                sil_df = pd.DataFrame(sil_results)
                sil_df.to_csv(
                    f"{save_path}/silhouette_E{epoch}_R{run_time}.csv",
                    index=False
                )

                print(f"[DONE] Epoch {epoch}, Run {run_time}, Name '{run_name_short}'")

