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

#full run 30000
def startimportdata(df,figure_path,model_path,data_path,name,thershold_count=30000):
    #thershold_count=30000
    print(df.columns)

    df = df.dropna()

    df = df.sort_values(by=["id", 'visits'])
    df = df.drop(columns=['PERSONID'])
    keep=['Sex','readmission','Omsorg', 'Avdeling','YearEpisode', 'Hastegrad','Pasienter','age_cat','ICD10_code_diagnosis','bidiagnosiscode','treatment_time','id','visits']
    df = df[keep]

    df["Sex"] = (
    df["Sex"]
    .astype(str)
    .str.strip()
    .str.upper()
    .map({"M": 0, "K": 1})
)


    enc = OrdinalEncoder()

    Feature_collums = [ 'readmission','Omsorg', 'Avdeling','YearEpisode', 'Hastegrad','Pasienter','age_cat','bidiagnosiscode','treatment_time']
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
    df.rename(columns={'ICD10_code_diagnosis':'nlabel'},inplace=True)


    print(df.columns)


    # %%
    # %%







    mask = df['nlabel'] != "Error diagnosis missing from rdap"
    df = df[mask]
    
    # Create subject-level labels for stratification

    counts = df['nlabel'].value_counts()
    mask = counts[counts >= thershold_count].index
    df = df[df['nlabel'].isin(mask)]

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["nlabel"])


    tranlate_dic = {}

    for i in df.index:

        if df.loc[i, 'label'] in tranlate_dic:
            pass
        else:
            tranlate_dic[str(df.loc[i, 'label'])] = df.loc[i, 'nlabel']
            #print(tranlate_dic[str(df.loc[i, 'label'])])
    # %%





    



    print(df['label'].nunique())

    # %%
    counts = df['nlabel'].value_counts()
    plt.figure(figsize=(12, 8))  # Adjust width if needed for longer labels
    # Plot a bar chart
    counts.plot(kind='bar', edgecolor='k')
    plt.xlabel("ICD-10 Group")
    plt.ylabel("Frequency")
    plt.title("Distribution of ICD-10 Groups")
    # Rotate x labels and align properly
    plt.xticks(rotation=45, ha='right')  # or rotation=90 for vertical labels

    plt.tight_layout()  # Adjusts layout to avoid clipping
    plt.savefig(os.path.join(figure_path,f"{name}_x_df_diag_label.png"), bbox_inches="tight", pad_inches=0.2)
    plt.show()
    plt.close()
   
    # %%

    print(counts)



    if "nlabel" in df.columns:
        df = df.drop('nlabel', axis=1)
    return df,tranlate_dic    


def startimportdata_re(df,figure_path,model_path,data_path,name):
    print(df.columns)

    df = df.dropna()

    df = df.sort_values(by=["id", 'visits'])
    df = df.drop(columns=['PERSONID'])
    keep=['Sex','readmission','Omsorg', 'Avdeling','YearEpisode', 'Hastegrad','Pasienter','age_cat','ICD10_code_diagnosis','bidiagnosiscode','treatment_time','id','visits']
    df = df[keep]

    enc = OrdinalEncoder()

    df["Sex"] = (
    df["Sex"]
    .astype(str)
    .str.strip()
    .str.upper()
    .map({"M": 0, "K": 1})
)


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
    df.rename(columns={'readmission':'label'},inplace=True)


    print(df.columns)

    admission_counts = df['id'].value_counts().sort_index()
    admission_stats = admission_counts.describe(percentiles=[.25, .5, .75, .95, .99])

    print(admission_stats)
    median = admission_counts.median()
    print("Median:", median)

    # Get the values
    admission_counts = df['id'].value_counts()
    n = len(admission_counts)
    mean = admission_counts.mean()
    std = admission_counts.std(ddof=1)  # sample std

    # 95% CI
    z = 1.96
    margin_error = z * (std / np.sqrt(n))
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error

    print(f"Mean: {mean:.2f}")
    print(f"95% Confidence Interval: ({ci_lower:.2f}, {ci_upper:.2f})")


    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.hist(admission_counts, bins=range(1, admission_counts.max() + 2), edgecolor='black')
    plt.title("Number of Admissions per Patient")
    plt.xlabel("Admissions")
    plt.ylabel("Number of Patients")
    plt.grid(True)
    plt.savefig(os.path.join(figure_path,f"{name}_subject_count.png"))
    plt.show()




    return df
