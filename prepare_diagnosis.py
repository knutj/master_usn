import pandas as pd
import matplotlib.pyplot as plt
df= pd.read_csv("data/cleaned_data.csv.gz")

df = df.dropna()
print(df['ICD10_code_diagnosis'].value_counts())
value_counts = df['ICD10_code_diagnosis'].value_counts()
remove_these = value_counts[value_counts>1000].index

mask = df['ICD10_code_diagnosis'].isin(remove_these)

df = df[mask]

print(df['ICD10_code_diagnosis'].value_counts())











diag = df['ICD10_code_diagnosis']
y_df = pd.DataFrame(diag)
y_df = y_df.rename(columns={'ICD10_code_diagnosis': 'label'})

counts = y_df['label'].value_counts()

# Plot a bar chart
counts.plot(kind='bar', edgecolor='k')
plt.xlabel("ICD-10 Group")
plt.ylabel("Frequency")
plt.title("Distribution of ICD-10 Groups")
plt.show()
plt.savefig('figures/y_diag_ni.png',bbox_inches="tight", pad_inches=0.2)











x_df =df.drop(["ICD10_code_diagnosis"],axis=1)
x_df =x_df.drop(["HovedDiagnosekode"],axis=1)
x_df =x_df.drop(["HovedDiagnose"],axis=1)
x_df =x_df.drop(["Bidiagnose"],axis=1)


x_df = x_df.drop(columns=['episode_start_datetime','episode_end_datetime','YearEpisode','Lokasjon','ICD10_code_diagnosis_group','prev_treatment_time','treatment_time'])

if "icd10_from_main" in x_df.columns:
    x_df = x_df.drop(columns=["icd10_from_main"])


print(x_df.columns)
x_df = x_df.drop(columns=['MndEpisode3','prev_episode_start','prev_episode_start','age','prev_id','age_group','EPISODESTARTTID','prev_episode_start','prev_readmission','age','EPISODESLUTTTID'])
x_df = x_df.drop(columns=['PERSONID'])
x_df = x_df.drop(columns=['pre_HovedDiagnosekode'])
if 'Kommune' in df.columns:
        x_df.drop(columns=['Kommune'])

print("write_xgo_files")
x_df.to_parquet('data/x_diag_df.parque',engine="pyarrow")
y_df.to_parquet('data/y_diag_df.parque',engine="pyarrow")
