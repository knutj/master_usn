# master_usn
The helper.py contains the trainer function that is used by both CNN-LSTM and LSTM
Oversample.py contains oversample_sequences_smotets that use Euclidean distance to oversample.This is the SMOTE oversample that is adopted to time series.
Imporcsv_version1.py imports and merges the Excel spreadsheet exported from RDAP.
Clean_data3.py will clean up this data. It will remove duplicated visits that happen at the same time due to the secondary diagrams being too long. It will add a readmission column based on the time between stats. If this duration is less than 30 days, readmossio is set to 1. It also changes the DICO code to the ICD-10 chapter label. In addition, it also adds the duration of stays and the visit number.
model.py contains two deep neural networks in PyTorch: CNN-LSMT and only LSTM, both of which use time series. 
prepare_diagnosis will make a file that is used by xgobost for diagnosis based on ICD10 chapter labels.
process_data will prepare the time series and will change the data form id, feature into id,timeserie, feature. If the number is less than the number of sequences set, it will pad the missing time series. It also contains a function to make a pretty classification report.
Xgo_diat_smote is the XGBost file that predicts chapter iCD10 based on the feature, and it uses ordinary Smote oversampling.
Xgo_v2_random uses a random undersampling of the majority class and Xgboost to predict readmission based on the features.
cnn_lsm_diagnosev_rdap_v3 uses a CNN-LSM model trained on time-series features to predict ICD-10 chapter labels. 
lsm_diagnosev_rdap_v3.py uses LSMT based on time-series features to predict ICD-10 chapter labels. 
cnn_lstm_v4_TSDv5.py uses a CNN-LSM model trained on time-series features to predict readmission 
