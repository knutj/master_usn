import os
#import all relevant libraries
#!pip install seaborn
import pip

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
#X = pd.read_parquet('data/x_df.parque')
#y = pd.read_parquet('data/y_df.parque')

df = pd.read_csv("data/cleaned_data.csv.gz")




# Define the class weight scale (a hyperparameter) as the ration of negative labels to positive labels.
# This instructs the classifier to address the class imbalance.
class_weight_scale = 1.*y.label.value_counts()[0]/y.label.value_counts()[1]
class_weight_scale = 1

figure_path=os.path.join(os.getcwd(), 'figures','xgo_re')
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

model_path=os.path.join(os.getcwd(), 'models', 'xgo_re')
if not os.path.exists(model_path):
    os.makedirs(model_path)

data_path=os.path.join(os.getcwd(), 'data', 'xgo_re')
if not os.path.exists(data_path):
    os.makedirs(data_path)


X = X.dropna()
y = y.loc[X.index]

if "Bidiagnose" in X.columns:
    X = X.drop(columns=['Bidiagnose'])


print(X.columns)
X = X.drop(columns=['id'])
    
#if "Kommune" in X.columns:
 #   X = X.drop(columns=['Kommune'])


if "Kommune" in X.columns:
    categorical_cols = ['Omsorg', 'Avdeling', 'Hastegrad', 'Sex',
                    'bidiagnosiscode', 'ICD10_code_diagnosis', 'age_cat','Kommune']
else:
    categorical_cols = ['Omsorg', 'Avdeling', 'Hastegrad', 'Sex',
                    'bidiagnosiscode', 'ICD10_code_diagnosis', 'age_cat']

print(X.columns)

#for col in ['Omsorg', 'Avdeling', 'Hastegrad', 'Sex','bidiagnosiscode','ICD10_code_diagnosis','age_cat']:
#    X[col] = X[col].astype('category')



# Create a dictionary mapping old names to new names
if "Kommune" in X.columns:
    column_rename = {
    'Omsorg': 'Care',
    'Avdeling': 'Department',
    'Hastegrad': 'Urgency',
    'Sex': 'Sex',
    'bidiagnosiscode': 'SecondaryDiagnosisCode',
    'ICD10_code_diagnosis': 'ICD10DiagnosisCode',
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
    'ICD10_code_diagnosis': 'ICD10DiagnosisCode',
    'age_cat': 'AgeCategory'
}
    













print("start encoder")
encoder = OrdinalEncoder()
X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# Rename the columns
X.rename(columns=column_rename, inplace=True)





print("finished encoder")

x_df_all, X_val, y_df_all, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

print("under samling")
rus = RandomUnderSampler(random_state=0)
#rus  = NearMiss(version=1)
x_df, y_df = rus.fit_resample(x_df_all, y_df_all)
print("findished under sampleing")


print(y_df_all.value_counts())           # Before undersampling
print(y_df.value_counts()) # After undersampling
for col in x_df.columns:
    print(col,x_df[col].dtype)


# Setting minimal required initial hyperparameters

param={
    'objective':'binary:logistic',
    'nthread':4,
    'scale_pos_weight':class_weight_scale,
    'seed' : 1,
    'device':'cpu'
}
    
xgb1 = XGBClassifier(enable_categorical = True)
xgb1.set_params(**param)

def interp(x_new, x_old, y_old):
    return np.interp(x_new, x_old, y_old)


#for col in x_df.columns:
#    print(f"Colummn {col}, Type: {x_df[col].dtype}")

#x_df = x_df.drop(columns=['age_cat'])   





for col in x_df.columns:
    print(f"Colummn {col}, Type: {x_df[col].dtype}")


y_df['label'] = y_df['label'].astype(int)

# Ensure your dataframes (x_df, y_df) are loaded here
# For example:
# x_df = pd.read_csv('path_to_x_data.csv')
# y_df = pd.read_csv('path_to_y_data.csv')
#import cupy as cp
#import cudf
# Convert specified columns to 'category' dtype for XGBoost categorical support
#categorical_columns = ['gender', 'marital_status', 'insurance']
#for col in categorical_columns:
#    if col in x_df.columns:
#        x_df[col] = x_df[col].astype('category')

# Initialize XGBClassifier with categorical support
xgb1 = XGBClassifier(enable_categorical=True, eval_metric='logloss', device = "cuda")

# Parameters for cross-validation and ROC plotting
K = 5
skf = StratifiedKFold(n_splits=K)
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
lw = 2
i = 0
roc_aucs_xgb1 = []

fig = plt.figure(figsize=(7, 7))

# Cross-validation loop
for train_indices, test_indices in skf.split(x_df, y_df['label']):
    X_train, y_train = x_df.iloc[train_indices], y_df['label'].iloc[train_indices]
    X_valid, y_valid = x_df.iloc[test_indices], y_df['label'].iloc[test_indices]

    

    # Compute class weight scale for imbalanced data
    class_weight_scale = 1. * y_train.value_counts()[0] / y_train.value_counts()[1]
    print('class weight scale : {}'.format(class_weight_scale))
    class_weight_scale = 1
    xgb1.set_params(scale_pos_weight=class_weight_scale)

    #dtrain = xgb.DMatrix(X_train, label=y_train, device='cuda')
    #dtest = xgb.DMatrix(X_test, label=y_test, device='cuda')
    # Train the model
    #X_train = cudf.DataFrame.from_pandas(X_train)
    #X_valid = cudf.DataFrame.from_pandas(X_valid)
    #y_train = cudf.DataFrame.from_pandas(y_train)
    #y_valid = cudf.DataFrame.from_pandas(y_valid)

    #dtrain = xgb.DMatrix(x_df,label=y_df['label'],enable_categorical=True)

    xgb1.fit(X_train,y_train)

    # Predict probabilities for ROC curve
    xgb1_pred_prob = xgb1.predict_proba(X_valid)
    fpr, tpr, thresholds = roc_curve(y_valid, xgb1_pred_prob[:, 1])

    # Use numpy.interp to interpolate TPR values
    mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0

    # Calculate AUC for this fold
    roc_auc = auc(fpr, tpr)
    roc_aucs_xgb1.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, label=f'ROC fold {i} (area = {roc_auc:0.2f})')

    i += 1

# Plot chance line
plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Luck')

# Compute mean ROC curve across folds
mean_tpr /= K
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
         label=f'Mean ROC (area = {mean_auc:0.2f})', lw=lw)

# Finalize plot
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Initial estimator ROC curve')
plt.legend(loc="lower right")

# Save the figure
fig.savefig(os.path.join(figure_path,'initial_ROC_rand.png'))
plt.show()  # Optionally display the plot
plt.close()

# Option to perform hyperparameter optimization. Otherwise loads pre-defined xgb_opt params
optimize = True
op = False
import psutil
import optuna

# Get logical CPU count (includes hyperthreads)
logical_cores = psutil.cpu_count(logical=True)

# Get physical core count
physical_cores = psutil.cpu_count(logical=False)

print("Logical cores (threads):", logical_cores)
print("Physical cores:", physical_cores)

X_train = x_df
y_train = y_df['label']
#X_train = cudf.DataFrame.from_pandas(X_train)

# Define the objective function for Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 250),
        'max_depth': trial.suggest_int('max_depth', 1, 6),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
        'eval_metric': 'logloss',
        'random_state': 42,
    }

    model = xgb.XGBClassifier(**params)
    auc = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=5, n_jobs=-1).mean()

    return auc

# Create study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Best parameters and best score
print("Best params:", study.best_params)
print("Best AUC:", study.best_value)

# Train final optimized model
xgb_opt = xgb.XGBClassifier(**study.best_params)
xgb_opt.fit(X_train, y_train)




if op:

    param_test0 = {
     'n_estimators':range(50,250,10)
    }
    print('performing hyperparamter optimization step 0')
    gsearch0 = GridSearchCV(estimator = xgb1, param_grid = param_test0, scoring='roc_auc',n_jobs=2, cv=5)
    gsearch0.fit(X_train,y_train)
    print(gsearch0.best_params_, gsearch0.best_score_)

    param_test1 = {
     'max_depth':range(1,4),
     'min_child_weight':range(1,10)
    }
    print('performing hyperparamter optimization step 1')
    gsearch1 = GridSearchCV(estimator = gsearch0.best_estimator_,
     param_grid = param_test1, scoring='roc_auc',n_jobs=2, cv=5)
    gsearch1.fit(X_train,y_train)
    print(gsearch1.best_params_, gsearch1.best_score_)

    max_d = gsearch1.best_params_['max_depth']
    min_c = gsearch1.best_params_['min_child_weight']

    param_test2 = {
     'gamma':[i/10. for i in range(0,5)]
    }
    print('performing hyperparamter optimization step 2')
    gsearch2 = GridSearchCV(estimator = gsearch1.best_estimator_,
     param_grid = param_test2, scoring='roc_auc',n_jobs=2, cv=5)
    gsearch2.fit(X_train,y_train)
    print(gsearch2.best_params_, gsearch2.best_score_)

    param_test3 = {
        'subsample':[i/10.0 for i in range(1,10)],
        'colsample_bytree':[i/10.0 for i in range(1,10)]
    }
    print('performing hyperparamter optimization step 3')
    gsearch3 = GridSearchCV(estimator = gsearch2.best_estimator_,
     param_grid = param_test3, scoring='roc_auc',n_jobs=4, cv=5)
    gsearch3.fit(X_train,y_train)
    print(gsearch3.best_params_, gsearch3.best_score_)

    param_test4 = {
        'reg_alpha':[0, 1e-5, 1e-3, 0.1, 10]
    }
    print('performing hyperparamter optimization step 4')
    gsearch4 = GridSearchCV(estimator = gsearch3.best_estimator_,
     param_grid = param_test4, scoring='roc_auc',n_jobs=4, cv=5)
    gsearch4.fit(X_train,y_train)
    print(gsearch4.best_params_, gsearch4.best_score_)

    alpha = gsearch4.best_params_['reg_alpha']
    if alpha != 0:
        param_test4b = {
            'reg_alpha':[0.1*alpha, 0.25*alpha, 0.5*alpha, alpha, 2.5*alpha, 5*alpha, 10*alpha]
        }
        print('performing hyperparamter optimization step 4b')
        gsearch4b = GridSearchCV(estimator = gsearch4.best_estimator_,
         param_grid = param_test4b, scoring='roc_auc',n_jobs=2, cv=5)
        gsearch4b.fit(X_train,y_train)
        print(gsearch4b.best_params_, gsearch4.best_score_)
        print('\nParameter optimization finished!')
        xgb_opt = gsearch4b.best_estimator_
        xgb_opt
    else:
        xgb_opt = gsearch4.best_estimator_
        xgb_opt
#else:
    # Pre-optimized settings
#    xgb_opt = XGBClassifier(base_score=0.5, colsample_bylevel=1.0, colsample_bytree=0.8,
#       gamma=0.1, learning_rate=0.1, max_delta_step=0, max_depth=7,
#       min_child_weight=9, n_estimators=220, nthread=4,subsample=0.9,
#       objective='binary:logistic', reg_alpha=5.0, reg_lambda=1,
#       scale_pos_weight=class_weight_scale, seed=1, enable_categorical = True)

print(xgb_opt)


K = 5
eval_size = int(np.round(1./K))
skf = StratifiedKFold(n_splits=K)


fig = plt.figure(figsize=(7,7))
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
lw = 2
i = 0
roc_aucs_xgbopt = []
for train_indices, test_indices in skf.split(x_df, y_df['label']):
    X_train, y_train = x_df.iloc[train_indices], y_df['label'].iloc[train_indices]
    X_valid, y_valid = x_df.iloc[test_indices], y_df['label'].iloc[test_indices]

    #X_train = cudf.DataFrame.from_pandas(X_train)
    #X_valid = cudf.DataFrame.from_pandas(X_valid)

    class_weight_scale = 1.*y_train.value_counts()[0]/y_train.value_counts()[1]
    print('class weight scale : {}'.format(class_weight_scale))
    class_weight_scale = 1
    xgb_opt.set_params(**{'scale_pos_weight' : class_weight_scale})
    xgb_opt.fit(X_train,y_train)
    xgb_opt_pred_prob = xgb_opt.predict_proba(X_valid)
    fpr, tpr, thresholds = roc_curve(y_valid, xgb_opt_pred_prob[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    roc_aucs_xgbopt.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    i += 1

plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
         label='Luck')

mean_tpr /= K
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")

fig.savefig(os.path.join(figure_path,'ROC_near.png'))
plt.show()

if op:

    aucs = [np.mean(roc_aucs_xgb1),
            gsearch0.best_score_,
            gsearch1.best_score_,
            gsearch2.best_score_,
            gsearch3.best_score_,
            gsearch4.best_score_,
            np.mean(roc_aucs_xgbopt)]

    fig = plt.figure(figsize=(4,4))
    plt.scatter(np.arange(1,len(aucs)+1), aucs)
    plt.plot(np.arange(1,len(aucs)+1), aucs)
    plt.xlim([0.5, len(aucs)+0.5])
    plt.ylim([0.99*aucs[0], 1.01*aucs[-1]])
    plt.xlabel('Hyperparamter optimization step')
    plt.ylabel('AUC')
    plt.title('Hyperparameter optimization')
    plt.grid()
    fig.savefig(os.path.join(figure_path,'optimization_rand.png'))

def my_plot_importance(booster, figsize, **kwargs):
    from matplotlib import pyplot as plt
    from xgboost import plot_importance
    fig, ax = plt.subplots(1,1,figsize=(figsize))
    plot_importance(booster=booster, ax=ax, **kwargs)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label,] +
ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(10)
    plt.tight_layout()
    fig.savefig(os.path.join(figure_path,'Feature_importance_rand.png'))

my_plot_importance(xgb_opt, (5,10))



from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix 
print(classification_report(y_true = y_df.label, y_pred = xgb_opt.predict(x_df)))
print(classification_report(y_true = y_df.label, y_pred = xgb_opt.predict(x_df)))
y_pred=xgb_opt.predict(X_val)


reports_dict=classification_report(y_df,xgb_opt.predict(x_df) ,output_dict=True)
report_df = pd.DataFrame(reports_dict).transpose()

# Save as CSV
report_df.to_csv(os.path.join(data_path,"classification_report_near_0.5_data.csv"))



print("validation set")
print(classification_report(y_true = y_val.label, y_pred = xgb_opt.predict(X_val)))
y_pred=xgb_opt.predict(X_val)


reports_dict=classification_report(y_val, y_pred = xgb_opt.predict(X_val),output_dict=True)
report_df = pd.DataFrame(reports_dict).transpose()

# Save as CSV
report_df.to_csv(os.path.join(data_path,"classification_report_rand_0.5.csv"))


training_data = confusion_matrix(y_df,xgb_opt.predict(x_df))
plt.figure(figsize=(8, 6))
sms.heatmap(training_data, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix trainingdata')
plt.savefig(os.path.join(figure_path,"trainingcm_rand.png"))
plt.show()






#https://stackoverflow.com/questions/76784223/how-to-plot-a-confusion-matrix
matrix = confusion_matrix(y_val, y_pred, normalize="pred")
disp = ConfusionMatrixDisplay(confusion_matrix=matrix) 
# Then just plot it: 
disp.plot() 
# And show it:
plt.savefig(os.path.join(figure_path,"cm_rand.png"))
plt.show()



cm = confusion_matrix(y_val, y_pred)

# 4. Display the confusion matrix nicely with seaborn
plt.figure(figsize=(8, 6))
sms.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(figure_path,"cm2_rand.png"))
plt.show()


# 5. Print a classification report
#print("Classification Report:")
#print(classification_report(true_labels, pred_labels))
#print(classification_report(true_labels, pred_labels, target_names=tranlate_dic.values()))



from sklearn.metrics import average_precision_score
preds = xgb_opt.predict_proba(X_val)[:, 1]
thresholds = np.arange(0.1, 0.9, 0.05)

#pr_auc = average_precision_score(y_true, preds)


scores = [average_precision_score(y_val, preds > t) for t in thresholds]
best_t = thresholds[np.argmax(scores)]
print(f"Precision-Recall AUC: {best_t:.2f}, Precision-Recall AUC: {max(scores):.3f}")
f=open(os.path.join(data_path,"thresholdsmoot.txt"),"w")
for t in thresholds:
    f.write(f"{t} and fscore {average_precision_score(y_val, preds > t)} \n")
f.close()


# Get predicted probabilities for class 1
y_probs = xgb_opt.predict_proba(X_val)[:, 1]

# Set your custom threshold
threshold = best_t

# Apply the threshold to get predicted labels
y_pred_custom = (y_probs > threshold).astype(int)

# Now use these predictions for metrics or confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_val, y_pred_custom))
print(confusion_matrix(y_val, y_pred_custom))

cm = confusion_matrix(y_val, y_pred_custom)
plt.figure(figsize=(8, 6))
sms.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(figure_path,"cm3_near.png"))
plt.show()


reports_dict=classification_report(y_val, y_pred_custom,output_dict=True)
report_df = pd.DataFrame(reports_dict).transpose()

# Save as CSV
report_df.to_csv(os.path.join(data_path,"classification_report_rand.csv"))


for col, cats in zip(categorical_cols, encoder.categories_):
    print(f"Column: {col}")
    for i, cat in enumerate(cats):
        print(f"  {i} → {cat}")
    print()

#import matplotlib.pyplot as plt
#xgb.plot_tree(xgb_opt,num_trees=0)
#plt.show()
print(xgb_opt.get_booster().get_dump()[0])    

import graphviz as gv

plt.figure(figsize=(30, 20))
xgb.plot_tree(xgb_opt, num_trees=0)
plt.savefig(os.path.join(figure_path,"re_rdap_tree.png"), bbox_inches="tight", dpi=300)
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
graph.render(os.path.join(figure_path,"re_rdap_tree_large.png"), format="png", cleanup=True)
