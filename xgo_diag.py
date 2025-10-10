import os
#import all relevant libraries
#!pip install seaborn
import pip
#import cudf
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import seaborn as sms
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, roc_curve, auc
from sklearn.preprocessing import OrdinalEncoder
from imblearn.under_sampling import RandomUnderSampler,NearMiss
from sklearn.model_selection import cross_val_score
#!pip install scipy
#from scipy import interp
from sklearn.model_selection import train_test_split

# Read preprocessed dataframes produced in notebook 2_preprocessing.ipynb
X = pd.read_parquet('data/x_diag_df.parque')
y = pd.read_parquet('data/y_diag_df.parque')


X = X.dropna()
y = y.loc[X.index]

y = y.rename(columns={'label':'dlabel'})

print(X.columns)
X = X.drop(columns=['id'])

# Create a dictionary mapping old names to new names
if "Kommune" in X.columns:
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

if "Kommune" in X.columns:
    categorical_cols = ['Omsorg', 'Avdeling', 'Hastegrad', 'Sex',
                    'bidiagnosiscode', 'age_cat','Kommune']
else:
    categorical_cols = ['Omsorg', 'Avdeling', 'Hastegrad', 'Sex',
                    'bidiagnosiscode', 'age_cat']
    

print("start encoder")
encoder = OrdinalEncoder()
X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# Rename the columns
X.rename(columns=column_rename, inplace=True)


counts = y['dlabel'].value_counts()
print(y['dlabel'].value_counts())
# Plot a bar chart
counts.plot(kind='bar', edgecolor='k')
plt.xlabel("ICD-10 Group")
plt.ylabel("Frequency")
plt.title("Distribution of ICD-10 Groups")

plt.savefig('figures/y_diag_ni.png',bbox_inches="tight", pad_inches=0.2)



from sklearn.preprocessing import LabelEncoder

tranlate_dic={}



le = LabelEncoder()

y["label"] = le.fit_transform(y["dlabel"])

for i in y.index:

    if y.loc[i, 'label'] in tranlate_dic:
        pass
    else:
        tranlate_dic[str(y.loc[i, 'label'])] = y.loc[i, 'dlabel']

print(tranlate_dic)
print("finished encoder")

x_df, X_val, y_df, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

#print("under samling")
#rus = RandomUnderSampler(random_state=0)
##rus  = NearMiss(version=1)
#x_df, y_df = rus.fit_resample(x_df_all, y_df_all)
print("findished under sampleing")
number_class = np.unique(y_df["label"])












param = {
    'objective': 'multi:softmax',  # or 'multi:softprob' for probability output
    'num_class': number_class,
    'nthread': 10,
    'seed': 1,
    'device': 'cuda'
}


xgb1 = XGBClassifier()
xgb1.set_params(**param)

def interp(x_new, x_old, y_old):
    return np.interp(x_new, x_old, y_old)

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
#import cudf
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Get the overall set of classes from y_df
classes = np.unique(y_df['label'])

# Prepare cross-validation parameters
K = 5
skf = StratifiedKFold(n_splits=K)
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
lw = 2
i = 0
roc_aucs_xgb1 = []
con_list = []
fig = plt.figure(figsize=(7, 7))





for train_indices, test_indices in skf.split(x_df, y_df['label']):
    X_train, y_train = x_df.iloc[train_indices], y_df['label'].iloc[train_indices]
    X_valid, y_valid = x_df.iloc[test_indices], y_df['label'].iloc[test_indices]

   
    #xgb1.fit(X_train, y_train,sample_weight=class_weight_scale)
    xgb1.fit(X_train, y_train)
    
    # Predict probabilities for all classes (shape: [n_valid, n_classes])
    xgb1_pred_prob = xgb1.predict_proba(X_valid)

    cm = confusion_matrix(xgb1.predict(X_valid), y_valid)
    con_list.append(cm)
    # Binarize the true labels for the current fold using all classes
    y_valid_bin = label_binarize(y_valid, classes=classes)

    # Compute micro-average ROC: flatten both true and predicted probabilities
    fpr, tpr, thresholds = roc_curve(y_valid_bin.ravel(), xgb1_pred_prob.ravel())
    roc_auc = auc(fpr, tpr)
    roc_aucs_xgb1.append(roc_auc)

    # Plot this fold's micro-average ROC curve
    plt.plot(fpr, tpr, lw=lw, label=f'ROC fold {i} (AUC = {roc_auc:0.2f})')

    # Interpolate TPR for mean ROC calculation
    mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    i += 1

# Plot chance line
plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Chance')

# Compute mean ROC curve across folds
mean_tpr /= K
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
         label=f'Mean ROC (AUC = {mean_auc:0.2f})', lw=lw)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Micro-Average ROC Curve (One-vs-Rest) for Multi-Class')
plt.legend(loc="lower right")
fig.savefig('figures/initial_ROC_diag.png')
plt.show()


for cm in con_list:
    #print(cm)
    # Visualize the confusion matrix using seaborn
    #cm = np.diag(cm)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


optimize = True

from sklearn.model_selection import cross_val_score

X_train = x_df

y_train = y_df['label']


print(type(X_train))


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 250),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
        'eval_metric': 'logloss',
        'random_state': 42,
        'objective': 'multi:softprob',  # or 'multi:softprob' for probability output
        'num_class': number_class,
        'nthread': 10,
        'seed': 1,
        'device': 'cuda'
    }
    model = xgb.XGBClassifier(**params)

    auc = cross_val_score(
        model, X_train, y_train,
        scoring='roc_auc_ovr',  # Specify explicitly 'ovr' for multi-class
        cv=5,
        n_jobs=-1
    ).mean()
    return auc

import optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Best parameters and best score
print("Best params:", study.best_params)
print("Best AUC:", study.best_value)

# Train final optimized model
xgb_opt = xgb.XGBClassifier(**study.best_params)
xgb_opt.fit(X_train, y_train)


print(xgb_opt)    


# Prepare cross-validation parameters
K = 5
skf = StratifiedKFold(n_splits=K)
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
lw = 2
i = 0
roc_aucs_xgb1 = []
con_list = []
fig = plt.figure(figsize=(7, 7))

for train_indices, test_indices in skf.split(x_df, y_df['label']):
    X_train, y_train = x_df.iloc[train_indices], y_df['label'].iloc[train_indices]
    X_valid, y_valid = x_df.iloc[test_indices], y_df['label'].iloc[test_indices]

    
    # Train the model
    
    xgb_opt.fit(X_train,y_train)
    # Predict probabilities for all classes (shape: [n_valid, n_classes])
    
    xgb_opt_pred_prob = xgb_opt.predict_proba(X_valid)

    cm = confusion_matrix(xgb_opt.predict(X_valid), y_valid)
    con_list.append(cm)
    # Binarize the true labels for the current fold using all classes
    y_valid_bin = label_binarize(y_valid, classes=classes)

    # Compute micro-average ROC: flatten both true and predicted probabilities
    fpr, tpr, thresholds = roc_curve(y_valid_bin.ravel(), xgb_opt_pred_prob.ravel())
    roc_auc = auc(fpr, tpr)
    roc_aucs_xgb1.append(roc_auc)

    # Plot this fold's micro-average ROC curve
    plt.plot(fpr, tpr, lw=lw, label=f'ROC fold {i} (AUC = {roc_auc:0.2f})')

    # Interpolate TPR for mean ROC calculation
    mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    i += 1

# Plot chance line
plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Chance')

# Compute mean ROC curve across folds
mean_tpr /= K
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
         label=f'Mean ROC (AUC = {mean_auc:0.2f})', lw=lw)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Micro-Average ROC Curve (One-vs-Rest) for Multi-Class')
plt.legend(loc="lower right")
fig.savefig('figures/ROC_diag.png')
plt.show()


for cm in con_list:
    #print(cm)
    # Visualize the confusion matrix using seaborn
    #cm = np.diag(cm)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()



print(classification_report(y_true = y_df.label, y_pred = xgb_opt.predict(x_df)))

def my_plot_importance(booster, figsize, **kwargs):
    from matplotlib import pyplot as plt
    from xgboost import plot_importance
    fig, ax = plt.subplots(1,1,figsize=(figsize))
    plot_importance(booster=booster, ax=ax, **kwargs)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label,] +
ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(10)
    plt.tight_layout()
    fig.savefig('figures/Feature_importance_diag.png')

my_plot_importance(xgb_opt, (5,10))


a=classification_report(y_true = y_df.label, y_pred = xgb_opt.predict(x_df), output_dict=True)
b={}

#for j in tranlate_dic:
#    print(tranlate_dic[j])



tranlate_dic = {str(k): v for k,v in tranlate_dic.items()}

print(a)
for key in a:
    if key in tranlate_dic:
        tmp =tranlate_dic[key]
        #print(tmp)
        b[tmp] = a[key]
        #print(key, a[key])

formatted_text = f"{'Label':<80} {'Precision':>10},{'Recall':>11}, {'F1-Score':>11}, {'Support':>11}\n"
for label, metrics in b.items():
    fixed_label = f"{label:<80}"
    formatted_text += (
        f"{fixed_label}: {metrics['precision']:10.3f}, {metrics['recall']:10.3f}, {metrics['f1-score']:10.3f}, {metrics['support']:10.0f}\n"
    )
print(formatted_text)


from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix
y_pred=xgb_opt.predict(X_val)








print("validation set")
a=classification_report(y_true = y_val.label, y_pred = xgb_opt.predict(X_val),output_dict=True)

b={}

#for j in tranlate_dic:
#    print(tranlate_dic[j])



for key in a:
    if key in tranlate_dic:
        tmp =tranlate_dic[key]
        #print(tmp)
        b[tmp] = a[key]
        #print(key, a[key])

formatted_text = f"{'Label':<80} {'Precision':>10},{'Recall':>11}, {'F1-Score':>11}, {'Support':>11}\n"
for label, metrics in b.items():
    fixed_label = f"{label:<80}"
    formatted_text += (
        f"{fixed_label}: {metrics['precision']:10.3f}, {metrics['recall']:10.3f}, {metrics['f1-score']:10.3f}, {metrics['support']:10.0f}\n"
    )
print(formatted_text)





y_pred=xgb_opt.predict(X_val)


training_data = confusion_matrix(y_val.label,xgb_opt.predict(X_val),normalize="pred")
import seaborn as sms
#print(training_data)
#training_data = np.diag(training_data)
plt.figure(figsize=(8, 6))
sms.heatmap(training_data, annot=True, fmt='.1f', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix trainingdata')
plt.savefig("figures/trainingcm_diag.png")
plt.show()


for k, v in tranlate_dic.items():
    print(f"{k}: {v}")



import graphviz as gv

plt.figure(figsize=(30, 20))
xgb.plot_tree(xgb_opt, num_trees=0)
plt.savefig("figures/diag_rdap_tree.png", bbox_inches="tight", dpi=300)
plt.show()


from graphviz import Digraph
graph = xgb.to_graphviz(
    xgb_opt,
    num_trees=0,
    size="40,30",
    condition_node_params={'fontsize':'16'},
    leaf_node_params={'fontsize':'16'}
)

# Save as PNG directly to a file
graph.render("figures/di_rdap_tree_large.png", format="png", cleanup=True)