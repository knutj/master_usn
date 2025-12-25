import os

from optuna.logging import set_verbosity, WARNING
import dask.dataframe as dd
#os.system('cls' if os.name == 'nt' else 'clear')
import json
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import OrdinalEncoder, label_binarize
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import SMOTE
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import List, Tuple
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from optuna.logging import set_verbosity, CRITICAL
import sys
from sklearn.metrics import f1_score, roc_auc_score
from tsaug import TimeWarp, Drift, Reverse, Quantize, AddNoise
# %%
import random


def oversample_sequences_multiclass(X, y, lengths, target_class_size=None, shuffle=True, random_state=None):
    """
    Oversample sequences to balance all classes to a specified size (default: max class size).

    Parameters:
        X (np.ndarray): Array of shape (n_samples, seq_len, n_features)
        y (np.ndarray): Class labels of shape (n_samples,)
        lengths (np.ndarray): True sequence lengths, shape (n_samples,)
        target_class_size (int): Target number of samples per class (default: max class count)
        shuffle (bool): Whether to shuffle the result
        random_state (int): For reproducibility

    Returns:
        X_balanced, y_balanced, lengths_balanced
    """
    if random_state is not None:
        np.random.seed(random_state)

    class_counts = Counter(y)
    classes = np.unique(y)

    if target_class_size is None:
        target_class_size = max(class_counts.values())

    X_list, y_list, lengths_list = [X], [y], [lengths]

    for cls in classes:
        count = class_counts[cls]
        if count < target_class_size:
            # Indices of the current class
            idx = np.where(y == cls)[0]
            n_to_sample = target_class_size - count
            resampled_idx = np.random.choice(idx, size=n_to_sample, replace=True)

            X_list.append(X[resampled_idx])
            y_list.append(y[resampled_idx])
            lengths_list.append(lengths[resampled_idx])

    # Combine all and shuffle
    X_bal = np.concatenate(X_list, axis=0)
    y_bal = np.concatenate(y_list, axis=0)
    lengths_bal = np.concatenate(lengths_list, axis=0)

    if shuffle:
        indices = np.arange(len(y_bal))
        np.random.shuffle(indices)
        X_bal = X_bal[indices]
        y_bal = y_bal[indices]
        lengths_bal = lengths_bal[indices]

    return X_bal, y_bal, lengths_bal

def oversample_sequences_multiclass_tsaug(
    X, y, lengths, 
    target_class_size=None, 
    shuffle=True, 
    random_state=None
):
    """
    Oversample sequences for multiclass problems using tsaug-based data augmentation.
    Balances all classes by generating synthetic sequences instead of plain duplication.

    Parameters:
        X (np.ndarray): shape (n_samples, seq_len, n_features)
        y (np.ndarray): class labels, shape (n_samples,)
        lengths (np.ndarray): true sequence lengths
        target_class_size (int): desired samples per class (default: max class size)
        shuffle (bool): whether to shuffle after augmentation
        random_state (int): reproducibility seed

    Returns:
        X_aug, y_aug, lengths_aug
    """
    if random_state is not None:
        np.random.seed(random_state)

    class_counts = Counter(y)
    classes = np.unique(y)

    if target_class_size is None:
        target_class_size = max(class_counts.values())

    X_list, y_list, lengths_list = [X], [y], [lengths]

    # Define tsaug augmenter pipeline
    augmenter = (
        TimeWarp(n_speed_change=3, max_speed_ratio=1.5)
        + Drift(max_drift=(0.05, 0.1))
        + AddNoise(scale=0.02)
    )

    for cls in classes:
        count = class_counts[cls]
        if count < target_class_size:
            idx = np.where(y == cls)[0]
            n_to_sample = target_class_size - count

            # Randomly pick some sequences from the minority class
            base_samples = X[idx]
            chosen_idx = np.random.choice(len(base_samples), size=n_to_sample, replace=True)
            to_augment = base_samples[chosen_idx]

            # tsaug expects shape (batch, time, features)
            augmented = augmenter.augment(to_augment)

            X_list.append(augmented)
            y_list.append(np.full(n_to_sample, cls))
            lengths_list.append(lengths[idx[chosen_idx]])

    # Combine all results
    X_aug = np.concatenate(X_list, axis=0)
    y_aug = np.concatenate(y_list, axis=0)
    lengths_aug = np.concatenate(lengths_list, axis=0)

    if shuffle:
        indices = np.arange(len(y_aug))
        np.random.shuffle(indices)
        X_aug = X_aug[indices]
        y_aug = y_aug[indices]
        lengths_aug = lengths_aug[indices]

    return X_aug, y_aug, lengths_aug

import numpy as np
from collections import Counter

try:
    from tslearn.metrics import cdist_dtw
    USE_DTW = True
except ImportError:
    from scipy.spatial.distance import cdist
    print("⚠️ tslearn not found. Falling back to Euclidean distance for SMOTE-TS.")
    USE_DTW = False


def euclidean_gpu(X):
    """
    Compute pairwise Euclidean distance using GPU if available.
    X must be a 2D NumPy array (N × features).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    # Compute squared norms
    norms = (X_t ** 2).sum(dim=1).reshape(-1, 1)

    # Broadcasting trick
    dist = norms - 2 * (X_t @ X_t.T) + norms.T
    dist = torch.sqrt(torch.clamp(dist, min=0.0))

    return dist.cpu().numpy()   # Return NumPy for compatibility



def smote_ts_dtw(X, y, target_class, k=5, n_new=100, use_dtw=False,max_samples=20000):
    """
    Generate synthetic time series samples for a given class using DTW or Euclidean-based interpolation.
    """
    idx_class = np.where(y == target_class)[0]
    X_class = X[idx_class]
    MAX_DTW_SAMPLES = 300 

    if len(X_class) > 300:
        USE_DTW = False


    if len(X_class) > max_samples:
        print(f"⚠ Downsampling class {target_class} from {len(X_class)} to {max_samples} for SMOTE")
        keep_idx = np.random.choice(len(X_class), max_samples, replace=False)
        X_class = X_class[keep_idx]
    else:
        print(f"📏 Using {len(X_class)} samples for SMOTE class {target_class}")

    # Compute distance matrix
    #if use_dtw and USE_DTW:
    #    print(f"📏 Using DTW distance for class {target_class}...")
    #    dist_matrix = cdist_dtw(X_class)
    try:
          # --------------------------------------------------------
    # ⭐ Flatten sequences for GPU Euclidean distance
    # --------------------------------------------------------
        X_flat = X_class.reshape(len(X_class), -1)
        dist_matrix = euclidean_gpu(X_flat)
    
    except:
        print(f"📏 Using Euclidean distance for class {target_class}...")
        dist_matrix = cdist(X_class.reshape(len(X_class), -1), X_class.reshape(len(X_class), -1))

    np.fill_diagonal(dist_matrix, np.inf)

    X_new = []
    y_new = []

    for _ in range(n_new):
        i = np.random.choice(len(X_class))
        nn_idx = np.argsort(dist_matrix[i])[:k]
        j = np.random.choice(nn_idx)
        lam = np.random.rand()

        # Interpolate between two similar sequences
        synthetic = X_class[i] + lam * (X_class[j] - X_class[i])
        X_new.append(synthetic)
        y_new.append(target_class)

    return np.array(X_new), np.array(y_new)


def oversample_sequences_smotets(X, y, lengths, target_ratio=0.5, k=5):
    """
    Oversample minority classes in a multiclass setting using SMOTE-TS (DTW or Euclidean).
    """
    X, y, lengths = np.asarray(X), np.asarray(y), np.asarray(lengths)
    assert len(X) == len(y) == len(lengths), \
        f"Input length mismatch: X={len(X)}, y={len(y)}, lengths={len(lengths)}"

    class_counts = Counter(y)
    #print(f"class_counts {class_counts}" )
    max_class_size = max(class_counts.values())

    X_aug, y_aug, lengths_aug = [X], [y], [lengths]

    for cls, count in class_counts.items():
        if count >= max_class_size * target_ratio:
            print(f"✅ Class {cls} already satisfies target ratio.")
            continue

        n_target = int(max_class_size * target_ratio)
        n_new = n_target - count

        print(f"🔄 Oversampling class {cls}: current={count}, target={n_target}, generating={n_new}")
        X_new, y_new = smote_ts_dtw(X, y, cls, k=k, n_new=n_new)

        # Assign synthetic sequence lengths based on median of that class
        lengths_class = lengths[y == cls]
        median_len = int(np.median(lengths_class))
        lengths_new = np.full(len(X_new), median_len)

        X_aug.append(X_new)
        y_aug.append(y_new)
        lengths_aug.append(lengths_new)

    # Combine original and synthetic data
    X_final = np.concatenate(X_aug, axis=0)
    y_final = np.concatenate(y_aug, axis=0)
    lengths_final = np.concatenate(lengths_aug, axis=0)

    assert len(X_final) == len(y_final) == len(lengths_final), \
        f"Inconsistent lengths after SMOTE-TS: X={len(X_final)}, y={len(y_final)}, len={len(lengths_final)}"

    print(f"✅ Final dataset size: {len(X_final)}")
    return X_final, y_final, lengths_final

import numpy as np
import torch
try:
    import faiss
except:
    print("can not import faiss")
from collections import Counter

def oversample_sequences_multiclass_tsmote_faiss(
    X, y, lengths,
    target_class_size=None,
    k_neighbors=5,
    shuffle=True,
    random_state=None,
    device=None
):
    """
    GPU-accelerated T-SMOTE using FAISS for nearest neighbor search.
    
    Parameters:
        X (np.ndarray): shape (n_samples, seq_len, n_features)
        y (np.ndarray): class labels
        lengths (np.ndarray): sequence lengths
        target_class_size (int): target number of samples per class (default=max)
        k_neighbors (int): neighbors for interpolation
        shuffle (bool): whether to shuffle result
        random_state (int): random seed
        device (torch.device): 'cuda' or 'cpu' (auto-detected if None)

    Returns:
        X_bal, y_bal, lengths_bal
    """
    if random_state is not None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_counts = Counter(y)
    classes = np.unique(y)
    if target_class_size is None:
        target_class_size = max(class_counts.values())

    X_list, y_list, lengths_list = [X], [y], [lengths]

    for cls in classes:
        X_cls = X[y == cls]
        n_samples, seq_len, n_features = X_cls.shape
        count = n_samples

        if count < target_class_size:
            n_to_sample = target_class_size - count

            # --- Flatten time series for neighbor search ---
            X_flat = X_cls.reshape(n_samples, -1).astype(np.float32)

            # --- Setup FAISS index ---
            if torch.cuda.is_available():
                res = faiss.StandardGpuResources()
                index = faiss.IndexFlatL2(X_flat.shape[1])  # L2 distance
                index = faiss.index_cpu_to_gpu(res, 0, index)
            else:
                index = faiss.IndexFlatL2(X_flat.shape[1])
            index.add(X_flat)

            # --- Generate synthetic samples ---
            synthetic_samples = []
            for _ in range(n_to_sample):
                i = np.random.randint(0, n_samples)
                x_i = X_cls[i]
                _, indices = index.search(X_flat[i].reshape(1, -1), k_neighbors)
                # Avoid self
                j = np.random.choice(indices[0][1:])
                x_j = X_cls[j]

                # Interpolate on GPU
                x_i_t = torch.tensor(x_i, dtype=torch.float32, device=device)
                x_j_t = torch.tensor(x_j, dtype=torch.float32, device=device)
                alpha = torch.rand(1, device=device)
                x_syn = alpha * x_i_t + (1 - alpha) * x_j_t

                # Add temporal Gaussian noise
                temporal_noise = torch.randn_like(x_syn) * 0.01
                x_syn = x_syn + temporal_noise

                synthetic_samples.append(x_syn.cpu().numpy())

            synthetic_samples = np.array(synthetic_samples)
            synthetic_labels = np.full(len(synthetic_samples), cls)
            synthetic_lengths = np.full(len(synthetic_samples), seq_len)

            X_list.append(synthetic_samples)
            y_list.append(synthetic_labels)
            lengths_list.append(synthetic_lengths)

    # --- Combine and shuffle ---
    X_bal = np.concatenate(X_list, axis=0)
    y_bal = np.concatenate(y_list, axis=0)
    lengths_bal = np.concatenate(lengths_list, axis=0)

    if shuffle:
        idx = np.arange(len(y_bal))
        np.random.shuffle(idx)
        X_bal, y_bal, lengths_bal = X_bal[idx], y_bal[idx], lengths_bal[idx]

    return X_bal, y_bal, lengths_bal

from sklearn.neighbors import NearestNeighbors
from collections import Counter
import numpy as np
import torch

def oversample_sequences_multiclass_tsmote_gpu(
    X, y, lengths,
    target_class_size=None,
    k_neighbors=5,
    shuffle=True,
    random_state=None,
    device=None,
    max_samples=10000
):
    """
    GPU-accelerated Temporal-oriented SMOTE (T-SMOTE).
    Uses PyTorch tensors for fast interpolation and temporal noise.
    """

    if random_state is not None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_counts = Counter(y)
    classes = np.unique(y)
    if target_class_size is None:
        target_class_size = max(class_counts.values())

    X_list, y_list, lengths_list = [X], [y], [lengths]

    for cls in classes:
        X_cls = X[y == cls]

        # Downsample large classes for memory safety
        if len(X_cls) > max_samples:
            print(f"⚠ Downsampling class {cls} from {len(X_cls)} to {max_samples} for T-SMOTE")
            keep_idx = np.random.choice(len(X_cls), max_samples, replace=False)
            X_cls = X_cls[keep_idx]
        else:
            print(f"📏 Using {len(X_cls)} samples for T-SMOTE class {cls}")

        n_samples, seq_len, n_features = X_cls.shape
        count = n_samples

        if count < target_class_size:
            n_to_sample = target_class_size - count

            # Neighbor search (CPU)
            X_flat = X_cls.reshape(n_samples, -1)
            nn = NearestNeighbors(n_neighbors=min(k_neighbors, n_samples))
            nn.fit(X_flat)

            synthetic_samples = []
            for _ in range(n_to_sample):
                i = np.random.randint(0, n_samples)
                x_i = X_cls[i]
                _, indices = nn.kneighbors(X_flat[i].reshape(1, -1))
                j = np.random.choice(indices[0][1:])  # avoid self
                x_j = X_cls[j]

                # Move to GPU tensors
                x_i_t = torch.tensor(x_i, dtype=torch.float32, device=device)
                x_j_t = torch.tensor(x_j, dtype=torch.float32, device=device)

                # Interpolation and noise on GPU
                alpha = torch.rand(1, device=device)
                x_syn = alpha * x_i_t + (1 - alpha) * x_j_t
                temporal_noise = torch.randn_like(x_syn) * 0.01
                x_syn = x_syn + temporal_noise

                synthetic_samples.append(x_syn.cpu().numpy())

            synthetic_samples = np.array(synthetic_samples)
            synthetic_labels = np.full(len(synthetic_samples), cls)
            synthetic_lengths = np.full(len(synthetic_samples), seq_len)

            X_list.append(synthetic_samples)
            y_list.append(synthetic_labels)
            lengths_list.append(synthetic_lengths)

        torch.cuda.empty_cache()  # free memory after each class

    # Combine and shuffle
    X_bal = np.concatenate(X_list, axis=0)
    y_bal = np.concatenate(y_list, axis=0)
    lengths_bal = np.concatenate(lengths_list, axis=0)

    if shuffle:
        indices = np.arange(len(y_bal))
        np.random.shuffle(indices)
        X_bal, y_bal, lengths_bal = X_bal[indices], y_bal[indices], lengths_bal[indices]

    return X_bal, y_bal, lengths_bal
    
    
    import numpy as np
import torch
from collections import Counter
from sklearn.neighbors import NearestNeighbors

def balance_sequences_hybrid_tsmote(
    X, y, lengths,
    undersample_target=None,
    oversample_target=None,
    k_neighbors=5,
    shuffle=True,
    random_state=None,
    device=None,
    max_samples=10000
):
    """
    Optimized Hybrid balancing for time series data.
    """
    # 1. Setup
    if random_state is not None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    # Detect device (keep purely numpy if CPU is preferred to avoid transfer overhead)
    use_cuda = False
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            use_cuda = True
        else:
            device = torch.device("cpu")
    elif device.type == 'cuda':
        use_cuda = True

    class_counts = Counter(y)
    classes = np.unique(y)
    print(f"📊 Original class counts: {class_counts}")

    # Set defaults based on distribution
    counts_vals = list(class_counts.values())
    if undersample_target is None:
        undersample_target = int(np.median(counts_vals))
    
    # Pre-allocate lists
    X_final_list, y_final_list, lengths_final_list = [], [], []

    # --- PROCESS CLASSES ---
    for cls in classes:
        # Filter class data
        mask = (y == cls)
        X_cls = X[mask]
        len_cls = lengths[mask]
        count = len(X_cls)
        
        # --- A. UNDERSAMPLING ---
        if count > undersample_target:
            print(f"⚠️ Undersampling class {cls}: {count} → {undersample_target}")
            keep_idx = np.random.choice(count, undersample_target, replace=False)
            X_cls = X_cls[keep_idx]
            len_cls = len_cls[keep_idx]
            count = undersample_target # Update count for oversampling logic check
        else:
            print(f"📏 Keeping class {cls}: {count} samples")

        # Add processed original data to final lists
        X_final_list.append(X_cls)
        y_final_list.append(np.full(len(X_cls), cls))
        lengths_final_list.append(len_cls)

        # --- B. OVERSAMPLING (Vectorized T-SMOTE) ---
        # Determine target count (default to max class count found so far or specified)
        target_count = oversample_target if oversample_target else max(counts_vals)
        
        if count < target_count:
            n_to_sample = target_count - count
            print(f"🧠 T-SMOTE Class {cls}: Generating +{n_to_sample} samples")

            # Safety: Cannot SMOTE with < 2 samples
            if count < 2:
                print(f"   ↳ Not enough samples for SMOTE (n={count}). Random duplicating instead.")
                rand_idx = np.random.choice(count, n_to_sample, replace=True)
                X_syn, len_syn = X_cls[rand_idx], len_cls[rand_idx]
            else:
                # 1. Fit Nearest Neighbors on flattened temporal data
                # Handle max_samples for NN fitting (speed optimization)
                if count > max_samples:
                    fit_idx = np.random.choice(count, max_samples, replace=False)
                    X_fit = X_cls[fit_idx]
                else:
                    X_fit = X_cls

                n_neighbors = min(k_neighbors, len(X_fit) - 1)
                nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
                nn.fit(X_fit.reshape(len(X_fit), -1))

                # 2. Select Parents (Vectorized)
                # Randomly pick base samples (X_i) from available data
                base_indices = np.random.choice(len(X_cls), n_to_sample, replace=True)
                X_base = X_cls[base_indices]
                len_base = len_cls[base_indices] # Inherit length from base

                # Find neighbors for these base samples
                # Note: We query NN for the base samples to find their specific neighbors
                # We flatten just for the query
                dists, ind = nn.kneighbors(X_base.reshape(n_to_sample, -1), return_distance=False)
                
                # Randomly pick one neighbor (X_j) for each base sample
                # ind has shape (n_to_sample, n_neighbors)
                neighbor_cols = np.random.randint(0, n_neighbors, size=n_to_sample)
                neighbor_indices_local = ind[np.arange(n_to_sample), neighbor_cols]
                
                # Map back if we subsampled for X_fit, otherwise direct map
                if count > max_samples:
                     # If we fit on a subset, neighbor_indices point to X_fit, not X_cls
                     # We must retrieve from X_fit
                     X_neighbor = X_fit[neighbor_indices_local]
                else:
                     X_neighbor = X_cls[neighbor_indices_local]

                # 3. Interpolate (PyTorch for GPU speed, or Numpy)
                if use_cuda:
                    # Move batch to GPU
                    t_base = torch.from_numpy(X_base).to(device).float()
                    t_neighbor = torch.from_numpy(X_neighbor).to(device).float()
                    
                    # Generate random alphas: shape (n_to_sample, 1, 1) for broadcasting
                    lambdas = torch.rand(n_to_sample, 1, 1, device=device)
                    
                    # Vectorized Interpolation
                    t_syn = t_base + lambdas * (t_neighbor - t_base)
                    
                    # Add Noise
                    noise = torch.randn_like(t_syn) * 0.01
                    t_syn += noise
                    
                    X_syn = t_syn.cpu().numpy()
                    del t_base, t_neighbor, t_syn, lambdas
                    torch.cuda.empty_cache()
                else:
                    # Numpy implementation
                    lambdas = np.random.rand(n_to_sample, 1, 1)
                    X_syn = X_base + lambdas * (X_neighbor - X_base)
                    X_syn += np.random.normal(0, 0.01, X_syn.shape)

                # 4. Masking (Critical for variable lengths)
                # Ensure synthetic data respects padding. If base sample had len 10,
                # synthetic sample must be 0 after index 10.
                seq_len = X.shape[1]
                mask_matrix = np.arange(seq_len)[None, :] < len_base[:, None]
                # Expand mask to features
                mask_matrix = np.repeat(mask_matrix[:, :, np.newaxis], X.shape[2], axis=2)
                X_syn = np.where(mask_matrix, X_syn, 0.0)
                
                len_syn = len_base

            X_final_list.append(X_syn)
            y_final_list.append(np.full(n_to_sample, cls))
            lengths_final_list.append(len_syn)

    # Combine
    X_bal = np.concatenate(X_final_list, axis=0)
    y_bal = np.concatenate(y_final_list, axis=0)
    lengths_bal = np.concatenate(lengths_final_list, axis=0)

    if shuffle:
        indices = np.arange(len(y_bal))
        np.random.shuffle(indices)
        X_bal = X_bal[indices]
        y_bal = y_bal[indices]
        lengths_bal = lengths_bal[indices]

    print(f"🏁 Final class distribution: {Counter(y_bal)}")
    return X_bal, y_bal, lengths_bal
    

