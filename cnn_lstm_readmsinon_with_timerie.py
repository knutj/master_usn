import pandas as pd

import numpy as np

import torch

import torch.nn as nn

import optuna

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, f1_score

from sklearn.preprocessing import OrdinalEncoder

from torch.utils.data import Dataset, DataLoader

from imblearn.over_sampling import SMOTE

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

 

import seaborn as sms

 

 

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, roc_curve, auc

from sklearn.preprocessing import OrdinalEncoder

from imblearn.under_sampling import RandomUnderSampler,NearMiss

from sklearn.model_selection import cross_val_score

#!pip install scipy

#from scipy import interp

from sklearn.model_selection import train_test_split

 

 

#df = pd.read_csv("data/cleaned_data.csv.gz", nrows=10000)
df = pd.read_csv("data/cleaned_data.csv.gz")
print(df.columns)

# Step 1: Sort data by patient and visit order

df = df.dropna()

df = df.sort_values(by=["id", 'visits'])
df = df.drop(columns=['PERSONID'])
keep=['readmission','Omsorg', 'Avdeling','YearEpisode', 'Hastegrad','Pasienter','age_cat','ICD10_code_diagnosis','bidiagnosiscode','treatment_time','id','visits']
df = df[keep]
# Step 2: Group into sequences (e.g., up to 5 visits per patient)
enc = OrdinalEncoder()

Feature_collums = [ 'readmission','Omsorg', 'Avdeling','YearEpisode', 'Hastegrad','Pasienter','age_cat','ICD10_code_diagnosis','bidiagnosiscode','treatment_time']
df[Feature_collums] = enc.fit_transform(df[Feature_collums])

# Create a dictionary mapping old names to new names
if "Kommune" in df.columns:
    column_rename = {
    'Omsorg': 'Care',
    'Avdeling': 'Department',
    'Hastegrad': 'Urgency',
    'Sex': 'Sex',
    'bidiagnosiscode': 'SecondaryDiagnosisCode',
    'age_cat': 'AgeCategory',
    'Kommune': 'Municipality'
}
else:
    column_rename = {
    'Omsorg': 'Care',
    'Avdeling': 'Department',
    'Hastegrad': 'Urgency',
    'Sex': 'Sex',
    'bidiagnosiscode': 'SecondaryDiagnosisCode',
    'age_cat': 'AgeCategory'
}

# Rename the columns
df.rename(columns=column_rename, inplace=True)


print(df.columns)

sequences = []

labels = []


df.rename(columns={'readmission':'label'},inplace=True)

df.sort_values(by=['id','visits'],inplace=True)

admission_counts = df['id'].value_counts().sort_index()

# Convert to DataFrame (optional, for display)
admission_stats = admission_counts.describe()
print(admission_stats)


# --- Step 3: Set parameters ---
sequence_length = 13
exclude_columns = ['id', 'visits', 'label']

# --- Step 4: Create sequences ---
sequences = []
labels = []
lengths = []  # new: stores true (unpadded) lengths
groups   = []
for id, group in df.groupby('id'):
    group = group.sort_values('visits')

    features = group.drop(columns=exclude_columns).values
    targets = group['label'].values

    num_admissions = len(group)
    if num_admissions >=1:
        for i in range(1, num_admissions):  # We need at least one prior admission to predict next
            start_idx = max(0, i - sequence_length)
            seq = features[start_idx:i]

            true_len = len(seq)  # actual number of valid admissions


            # Padding if needed
            if len(seq) < sequence_length:
                padding = np.zeros((sequence_length - len(seq), features.shape[1]))
                seq = np.vstack((padding, seq))
            sequences.append(seq)
            labels.append(targets[i])  # Label for the i-th admission
            lengths.append(true_len)
            groups.append(subject_id)

# --- Step 5: Convert to numpy arrays ---
X = np.array(sequences)
y = np.array(labels)
lengths = np.array(lengths)
groups = np.array(groups)

values, counts = np.unique(y, return_counts=True)

# Print nicely
for v, c in zip(values, counts):
    print(f"Value {v}: {c} samples")
 

class ReadmissionDataset(torch.utils.data.Dataset):
    def __init__(self, X, y,lengths):
        self.X = torch.tensor(X, dtype=torch.float32)  # shape: (N, T, F)
        self.y = torch.tensor(y, dtype=torch.long)     # long for classification
        self.lengths = lengths
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_seq = self.X[idx]
        y_val = self.y[idx]
        lenghts = self.lengths[idx]
        return x_seq, lenghts, x_seq, y_val  # return twice for LSTM and CNN input
    
X_train_all, X_test, y_train_all, y_test, lengths_train_all, lengths_test, groups_train, groups_test = train_test_split(
    X, y, lengths, groups,
    test_size=0.2,
    random_state=42,
    stratify=y  # optional: preserves label ratio
)


test_data = ReadmissionDataset(X_test, y_test,lengths_test)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class LSTMStack(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMStack, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

    def forward(self, x,lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, (hn, cn) = self.lstm(packed)
        return hn[-1]  # (batch_size, hidden_size)  <-- 2D

class CNNExtractor(nn.Module):
    def __init__(self, input_size, cnn_channels, kernel_size, dropout):
        super(CNNExtractor, self).__init__()
        self.input_size = input_size
        self.cnn_channels = cnn_channels
        padding = kernel_size // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),

            nn.Conv1d(cnn_channels, cnn_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),

            nn.Dropout(dropout)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x expected as (B, S, F); Conv1d wants (B, F, S)
        x = x.permute(0, 2, 1)              # (B, F, S)
        x = self.cnn(x)                     # (B, C, S)
        # Collapse the time dimension so we return 2D features
        x = self.global_pool(x)
        x = x.squeeze(-1)

        return x  # (batch_size, cnn_channels)  <-- 2D

class DiagnosisModel(nn.Module):
    def __init__(self, input_size, hidden_size, cnn_channels, kernel_size, num_layers, dropout):
        super(DiagnosisModel, self).__init__()
        self.lstm = LSTMStack(input_size, hidden_size, num_layers)
        self.cnn = CNNExtractor(input_size, cnn_channels, kernel_size, dropout)

        in_features = hidden_size + cnn_channels  # match what we actually concatenate
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # 2 classe
        )

    def forward(self, lstm_input,lengths, cnn_input):
        lstm_features = self.lstm(lstm_input,lengths)   # (B, hidden_size)
        cnn_features = self.cnn(cnn_input)      # (B, cnn_channels)

        # Both branches are 2D now, so concatenation works
        combined = torch.cat([lstm_features, cnn_features], dim=1)  # (B, hidden_size + cnn_channels)
        return self.mlp(combined)  # (batch_size, 2)
    
from sklearn.model_selection import StratifiedKFold
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------------------
# Optuna Objective Function
# ------------------------------
enc = SMOTE(random_state=42)
#modeltype = "LSTM"
from sklearn.model_selection import StratifiedGroupKFold
modeltype = ["CNNLST"]
def objective(trial):
    import torch
    import torch.nn as nn

    hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
    num_layers = trial.suggest_int("num_layers", 1, 2)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    input_size=X.shape[1]


    kernel_size = trial.suggest_int("kernel_size", 1, 3)
    cnn_channels = trial.suggest_int("cnn_channels", 16, 64)
    model = DiagnosisModel(input_size=X.shape[2], hidden_size=hidden_dim,cnn_channels=cnn_channels,kernel_size=kernel_size ,num_layers=num_layers,dropout=dropout)
    #print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    aucs = []

    for train_index, val_index in skf.split(X_train_all, y_train_all,groups=groups_train):
        X_train, X_val = X_train_all[train_index], X_train_all[val_index]
        y_train, y_val = y_train_all[train_index], y_train_all[val_index]
        len_train, len_val = lengths_train_all[train_index], lengths_train_all[val_index]
        #print(y_train[0])
        #x_flat = X_train.reshape(X_train.shape[0], -1)
        #X_resampled, y_train = enc.fit_resample(x_flat, y_train)
        #X_train = X_resampled.reshape(-1, X_train.shape[1], X_train.shape[2])
        if X_val.shape[0] == 0 or X_train.shape[0] == 0:
            print(f"⚠️ Skipping fold {fold} due to empty train or val set.")
            continue



        train_data = ReadmissionDataset(X_train, y_train,len_train)
        val_data = ReadmissionDataset(X_val, y_val,len_val)

        train_loader = DataLoader(train_data, batch_size=64, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

        for epoch in range(5):  # short training for optimization
            model.train()
            for X_batch,length_batch,lstm_batch, y_batch in train_loader:
                lstm_batch = lstm_batch.to(device)
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                preds = model(lstm_batch,length_batch,X_batch)
                loss = criterion(preds, y_batch.long())
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        all_labels = []
        all_pos_scores = []
        with torch.no_grad():
            for X_batch,length_batch,lstm_batch, y_batch in val_loader:
                lstm_batch = lstm_batch.to(device)
                X_batch = X_batch.to(device)
                logits = model(lstm_batch,length_batch, X_batch)              # (B, 2), raw logits
                probs_pos = torch.softmax(logits, dim=1)[:, 1]          # (B,), P(class=1
                all_pos_scores.append(probs_pos.detach().cpu())
                all_labels.append(y_batch.detach().cpu())
        import torch
        y_true = torch.cat(all_labels,dim=0).numpy()                      # shape (N,)
        y_score = torch.cat(all_pos_scores,dim=0).numpy()                 # shape (N,)

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_score)                        # binary ROC AUC
        aucs.append(auc)

    mean_auc = float(np.mean(auc))
    return mean_auc    



study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Best trial:")
print(study.best_trial)

import optuna.visualization as vis

# Requires: pip install optuna[visualization]
try:
    fig1 = vis.plot_optimization_history(study)
    fig1.write_html("optuna_optimization_history.html")
    fig1.write_image("optuna_optimization_history.png")

    fig2 = vis.plot_param_importances(study)
    fig2.write_html("optuna_param_importances.html")
    fig2.write_image("optuna_param_importances.png")

    fig3 = vis.plot_slice(study)
    fig3.write_html("optuna_slice.html")
    fig3.write_image("optuna_slice.png")

    fig4 = vis.plot_parallel_coordinate(study)
    fig4.write_html("optuna_parallel_coordinate.html")
    fig4.write_image("optuna_parallel_coordinate.png")

    print("✅ Visualizations saved as HTML and PNG.")

except Exception as e:
    print("⚠️ Visualization export failed.")
    print(e)

def build_model(params):
    model = DiagnosisModel(
            input_size=X.shape[2],
            cnn_channels=params["cnn_channels"],
            kernel_size=params["kernel_size"],
            hidden_size=params["hidden_dim"],
            num_layers=params["num_layers"],
            dropout=params["dropout"]
        )
    return model

best_params = study.best_trial.params
print("Best parameters:", best_params)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(best_params).to(device)    


lr=best_params["lr"]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


aucs = []

for train_index, val_index in skf.split(X_train_all, y_train_all,groups_train):
    X_train, X_val = X_train_all[train_index], X_train_all[val_index]
    y_train, y_val = y_train_all[train_index], y_train_all[val_index]
    len_train, len_val = lengths_train_all[train_index], lengths_train_all[val_index]


    #x_flat = X_train.reshape(X_train.shape[0], -1)
    #X_resampled, y_train = enc.fit_resample(x_flat, y_train)
    #X_train = X_resampled.reshape(-1, X_train.shape[1], X_train.shape[2])




    train_data = ReadmissionDataset(X_train, y_train,len_train)
    val_data = ReadmissionDataset(X_val, y_val,len_val)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    for epoch in range(5):  # short training for optimization
        model.train()
        for X_batch,length_batch,lstm_batch, y_batch in train_loader:
            lstm_batch = lstm_batch.to(device)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(lstm_batch,length_batch,X_batch)
            loss = criterion(preds, y_batch.long())
            loss.backward()
            optimizer.step()

 # Evaluate
    model.eval()
    all_labels = []
    all_pos_scores = []
    with torch.no_grad():
        for X_batch,length_batch,lstm_batch, y_batch in val_loader:
            lstm_batch = lstm_batch.to(device)
            X_batch = X_batch.to(device)
            logits = model(lstm_batch,length_batch, X_batch)              # (B, 2), raw logits
            probs_pos = torch.softmax(logits, dim=1)[:, 1]          # (B,), P(class=1
            all_pos_scores.append(probs_pos.detach().cpu())
            all_labels.append(y_batch.detach().cpu())
    import torch
    if len(all_labels) == 0 or len(all_pos_scores) == 0:
        print("⚠️ Validation produced no batches. Returning score = 0.0")
        continue
    y_true = torch.cat(all_labels,dim=0).numpy()                      # shape (N,)
    y_score = torch.cat(all_pos_scores,dim=0).numpy()                 # shape (N,)

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_score)                        # binary ROC AUC
    aucs.append(auc)

mean_auc = float(np.mean(auc))
print(f"Mean 5-fold ROC-AUC: {mean_auc:.4f}")


model.eval()
all_labels = []
all_pos_scores = []
with torch.no_grad():
    for X_batch,length_batch,lstm_batch, y_batch in test_loader:
        lstm_batch = lstm_batch.to(device)
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(lstm_batch,length_batch, X_batch)              # (B, 2), raw logits
        probs_pos = torch.softmax(logits, dim=1)[:, 1]
        all_pos_scores.append(probs_pos.detach().cpu())
        all_labels.append(y_batch.detach().cpu().long())

y_true = torch.cat(all_labels,dim=0).numpy()                      # shape (N,)
y_score = torch.cat(all_pos_scores,dim=0).numpy()                 # shape (N,)
print("Test ROC AUC:", roc_auc_score(y_true, y_score))

# after you've filled all_labels (list of tensors) and all_pos_scores (list of tensors)
y_true  = torch.cat(all_labels, dim=0).cpu().numpy()        # shape (N,)
y_score = torch.cat(all_pos_scores, dim=0).cpu().numpy()    # shape (N,)

# turn probabilities into predicted labels (use a clear threshold, not round)
y_pred = (y_score >= 0.5).astype(np.int64)

print("Test F1:", f1_score(y_true, y_pred))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
y_true  = torch.cat(all_labels, dim=0).cpu().numpy()      # (N,)
y_score = torch.cat(all_pos_scores, dim=0).cpu().numpy()  # (N,)

# Pick a threshold (0.5, or your tuned one)
thr = 0.5
y_pred = (y_score >= thr).astype(np.int64)

# Confusion matrix (counts)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
print("Confusion matrix (counts):\n", cm)

# Normalized per true class (i.e., recall per class)
cm_norm = confusion_matrix(y_true, y_pred, labels=[0, 1], normalize="true")
print("Confusion matrix (normalized by true class):\n", cm_norm)

# Quick report (precision/recall/F1 per class)
print("\nClassification report:\n", classification_report(y_true, y_pred, digits=4))

# Optional: plot (if you’re in a notebook)
import matplotlib.pyplot as plt
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(values_format='d')
plt.title(f'Confusion Matrix (threshold={thr})')
plt.show()
