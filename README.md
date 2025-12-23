# master_usn
## 📁 Modules Summary

| File                          | Description |
|-------------------------------|-------------|
| `helper.py`                  | Defines the `trainer()` and 'evalute()' used in CNN-LSTM and LSTM training. |
| `oversample.py`              | SMOTE for time series using Euclidean distance. |
| `importcsv_version1.py`      | Loads and merges RDAP Excel files. |
| `clean_data3.py`             | Cleans data, adds readmission column, maps daignosis → ICD-10, calculates stay duration. |
| `model.py`                   | Contains CNN-LSTM and LSTM architectures in PyTorch. |
| `prepare_diagnosis.py`       | Prepares data for XGBoost to predict ICD-10 chapters. |
| `process_data.py`            | Transforms and pads time series data; includes classification reporting. |
| `xgo_diat_smote.py`          | XGBoost + SMOTE for ICD-10 chapter prediction. |
| `xgo_v2_random.py`           | XGBoost + random undersampling for readmission prediction. |
| `cnn_lstm_diagnosev_rdap_v3.py` | CNN-LSTM model for diagnosis prediction. |
| `lsm_diagnosev_rdap_v3.py`   | LSTM model for diagnosis prediction. |
| `cnn_lstm_v4_TSDv6.py`       | CNN-LSTM model for predicting readmission. |
| lstm_v4_TSDv2.py             |  LSTM model for predicting readmission. |
| sklean_logistic.py           |  Logistic regression from sklearn for predicting readmission. |
| dataclass.py                 |   Data class that is used to define the data class |
| data_import.py               |  Set up import of data and set treshold for icd10 labels |
