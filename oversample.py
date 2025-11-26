import os
import random
from tsaug import AddNoise, Drift, TimeWarp,Drift, Reverse, Quantize, AddNoise
from collections import Counter
import numpy as np
import torch

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



def smote_ts_dtw(X, y, target_class, k=5, n_new=100, use_dtw=False,max_samples=1000):
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


def oversample_sequences_smotets(X, y, lengths, target_ratio=0.1, k=5):
    """
    Oversample minority classes in a multiclass setting using SMOTE-TS (DTW or Euclidean).
    """
    X, y, lengths = np.asarray(X), np.asarray(y), np.asarray(lengths)
    assert len(X) == len(y) == len(lengths), \
        f"Input length mismatch: X={len(X)}, y={len(y)}, lengths={len(lengths)}"

    class_counts = Counter(y)
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


