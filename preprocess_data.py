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
    #y_df = y_df.astype(int)
    x_df = x_df.copy()
    y_df["label"] = x_df["label"]   # Ensure label is merged

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

#(X, y, lengths,groups) =  preprocess_data(x_df, y_df,sequence_length,exclude_columns)
#%%
def prepare_data(x__df: pd.DataFrame,y_df=None,sequence_length=9 ):
    if y_df is None and 'label' in x__df.columns:
        y_df = x_df[['label']]
    elif 'label' not in x__df.columns:
        x_df = x__df.copy()
        x_df['label'] = y_df.values
    
    x_df=x_df.sort_values(by=['id', 'visits'])

    # --- Step 3: Set parameters ---

    exclude_columns = ['id', 'visits', 'label']

    # --- Step 4: Create sequences ---
    sequences = []
    labels = []

    (X, y, lengths,groups)=preprocess_data(x_df, y_df, sequence_length, exclude_columns)
    return X, y, lengths, groups