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
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import os 
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix 
import seaborn as sms
# Read preprocessed dataframes produced in notebook 2_preprocessing.ipynb
X = pd.read_parquet('data/x_df.parque')
y = pd.read_parquet('data/y_df.parque')

# Define the class weight scale (a hyperparameter) as the ration of negative labels to positive labels.
# This instructs the classifier to address the class imbalance.
class_weight_scale = 1.*y.label.value_counts()[0]/y.label.value_counts()[1]
class_weight_scale = 1

figure_path=os.path.join(os.getcwd(), 'figures','logstic')
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

model_path=os.path.join(os.getcwd(), 'models', 'logistic')
if not os.path.exists(model_path):
    os.makedirs(model_path)

data_path=os.path.join(os.getcwd(), 'data', 'logsitc')
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

def interp(x_new, x_old, y_old):
    return np.interp(x_new, x_old, y_old)


model = LogisticRegression()


for col in x_df.columns:
    print(f"Colummn {col}, Type: {x_df[col].dtype}")


y_df['label'] = y_df['label'].astype(int)

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
    

    #dtrain = xgb.DMatrix(X_train, label=y_train, device='cuda')
    #dtest = xgb.DMatrix(X_test, label=y_test, device='cuda')
    # Train the model
    #X_train = cudf.DataFrame.from_pandas(X_train)
    #X_valid = cudf.DataFrame.from_pandas(X_valid)
    #y_train = cudf.DataFrame.from_pandas(y_train)
    #y_valid = cudf.DataFrame.from_pandas(y_valid)

    #dtrain = xgb.DMatrix(x_df,label=y_df['label'],enable_categorical=True)

    model.fit(X_train, y_train)

    # Predict probabilities for ROC curve
    y_pred = model.predict(X_valid)
    fpr, tpr, thresholds = roc_curve(y_valid, y_pred)

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

import psutil
import optuna

# Get logical CPU count (includes hyperthreads)
logical_cores = psutil.cpu_count(logical=True)

# Get physical core count
physical_cores = psutil.cpu_count(logical=False)

print("Logical cores (threads):", logical_cores)
print("Physical cores:", physical_cores)

# Define the objective function for Optuna
def objective(trial):


    iter=trial.suggest_int('max_iter',50,150)

    model = LogisticRegression(max_iter=iter)
    auc = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=5, n_jobs=-1).mean()

    return auc


# Create study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# Best parameters and best score
print("Best params:", study.best_params)
print("Best AUC:", study.best_value)
best_params = study.best_trial.params
bestiter=best_params['max_iter']
model = LogisticRegression(max_iter=bestiter)
model.fit(X_train, y_train)



print("validation set")
print(classification_report(y_true = y_val.label, y_pred = model.predict(X_val)))
y_pred=model.predict(X_val)


reports_dict=classification_report(y_val, y_pred = model.predict(X_val),output_dict=True)
report_df = pd.DataFrame(reports_dict).transpose()

# Save as CSV
report_df.to_csv(os.path.join(data_path,"classification_report_rand_0.5.csv"))


training_data = confusion_matrix(y_df,model.predict(x_df))
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
preds = model.predict_proba(X_val)[:, 1]
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
y_probs = model.predict_proba(X_val)[:, 1]

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
