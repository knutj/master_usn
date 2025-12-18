import pandas as pd
import numpy as np 
from typing import List, Tuple
from sklearn.metrics import *
def preprocess_data(
        x_df: pd.DataFrame,
        y_df: pd.DataFrame,
        sequence_length: int,
        exclude_columns: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepares padded sequential data for LSTM input from admission records.

    Args:
        x_df (pd.DataFrame): Feature dataframe containing 'subject_id' and 'admittime'.
        y_df (pd.DataFrame): Label dataframe (should align with x_df).
        sequence_length (int): Max number of admissions in a sequence.
        exclude_columns (List[str]): Columns to exclude from features.

    Returns:
        Tuple:
            sequences (np.ndarray): Array of shape (samples, sequence_length, num_features).
            labels (np.ndarray): Labels for each sequence.
            lengths (np.ndarray): Actual length of each sequence before padding.
            groups (np.ndarray): Patient ID (subject_id) for each sequence.
    """
    y_df = y_df.astype(int)
    x_df = x_df.copy()
    y_df["label"] = x_df["label"]  # Ensure label is merged

    sequences = []
    labels = []
    lengths = []
    groups = []

    for subject_id, group in x_df.groupby("id"):
        group = group.sort_values("visits")

        features = group.drop(columns=exclude_columns).values
        targets = group["label"].values
        num_admissions = len(group)

        if num_admissions >= 1:
            for i in range(1, num_admissions):
                start_idx = max(0, i - sequence_length)
                seq = features[start_idx:i]

                true_len = len(seq)

                # Pad sequence if needed
                if true_len < sequence_length:
                    padding = np.zeros((sequence_length - true_len, seq.shape[1]))
                    seq = np.vstack((padding, seq))

                sequences.append(seq)
                labels.append(targets[i])
                lengths.append(true_len)
                groups.append(subject_id)

    return (
        np.array(sequences),
        np.array(labels),
        np.array(lengths),
        np.array(groups)
    )


# %%
def prepare_data(x__df: pd.DataFrame, y__df: pd.DataFrame, sequence_length=9):
    x_df = x__df.copy()
    y_df = y__df.copy()

    y_df = y_df.astype(int)
    bool_cols = x_df.select_dtypes(include='bool').columns
    x_df[bool_cols] = x_df[bool_cols].astype(int)

    # Merge labels with features
    x_df['label'] = y_df['label']

    # --- Step 2: Sort by patient and admission time ---
    x_df = x_df.sort_values(by=['id', 'visits'])

    # --- Step 3: Set parameters ---

    exclude_columns = ['id', 'visits', 'label']

    # --- Step 4: Create sequences ---
    sequences = []
    labels = []

    

    (X, y, lengths, groups) = preprocess_data(x_df, y_df, sequence_length, exclude_columns)
    return X, y, lengths, groups

def pretty_classification_report(y_true, y_pred, tranlate_dic, digits=4):
    """
    Generates a formatted classification report with translated class labels.

    Parameters:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        tranlate_dic (dict): Dictionary mapping original label keys to descriptive names
        digits (int): Number of digits for formatting (default: 4)

    Returns:
        str: Formatted text summary of the classification report
    """
    report = classification_report(y_true, y_pred, digits=digits, output_dict=True)
    report_with_name = {}

    for key in report:
        if key in tranlate_dic:
            label_name = tranlate_dic[key]
            report_with_name[label_name] = report[key]

    formatted_text = f"{'Label':<80} {'Precision':>10}, {'Recall':>10}, {'F1-Score':>10}, {'Support':>10}\n"
    formatted_text += "-" * 120 + "\n"

    for label, metrics in report_with_name.items():
        formatted_text += (
            f"{label:<80}: {metrics['precision']:10.3f}, {metrics['recall']:10.3f}, "
            f"{metrics['f1-score']:10.3f}, {metrics['support']:10.0f}\n"
        )

    return formatted_text,report_with_name