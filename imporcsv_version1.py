import dask.dataframe as dd
import pandas as pd
import os 

n=250
# Read CSV files

print(os.getcwd())
cwd = os.getcwd()
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))



excelfilename1 = os.path.join(cwd,'Diagnoser_21_2018_bygget_Pasid.xlsx')
excelfilename2 = os.path.join(cwd,'Diagnoser_25_22_bygget_Pasid.xlsx')

df1 = pd.read_excel(excelfilename1, sheet_name=0)
print(type(df1))
print(df1.columns)




df2 = pd.read_excel(excelfilename1,header=None, sheet_name=1)


df3 = pd.read_excel(excelfilename2, sheet_name=0)



df4 = pd.read_excel(excelfilename2, sheet_name=1,header=None)


# Use df1 as header source
headers = df1.columns
df2.columns = headers
df4.columns = headers

df1 = df1.drop(columns=['Kommune', 'KOMMUNEID'])

# Drop unwanted column if it exists
if "Unnamed: 0" in df1.columns:
    df1 = df1.drop(columns=["Unnamed: 0","Pasienter"])

df2 = df2.drop(columns=['Kommune', 'KOMMUNEID'])

# Drop unwanted column if it exists
if "Unnamed: 0" in df2.columns:
    df2 = df2.drop(columns=["Unnamed: 0","Pasienter"])

df3 = df3.drop(columns=['Kommune', 'KOMMUNEID'])

# Drop unwanted column if it exists
if "Unnamed: 0" in df3.columns:
    df3 = df3.drop(columns=["Unnamed: 0","Pasienter"])

df4 = df4.drop(columns=['Kommune', 'KOMMUNEID'])

# Drop unwanted column if it exists
if "Unnamed: 0" in df4.columns:
    df4 = df4.drop(columns=["Unnamed: 0","Pasienter"])


#df2 = df2.astype(dtypes)
#df4 = df4.astype(dtypes)
# Print headers
print("df1:", df1.columns)
print("df2:", df2.columns)
print("df3:", df3.columns)
print("df4:", df4.columns)

# Combine everything
df = pd.concat([df1, df2, df3, df4], axis=0)


df = df.dropna(subset=['PERSONID'])
# Optional cleanup
del df1, df2, df3, df4




# Rename problematic column
if "=class(AlderEpi,5)" in df.columns:
    df = df.rename(columns={'=class(AlderEpi,5)': 'age'})




for col in df.columns:
    print(f"{col}: {df[col].dtype}")


#remove kommune and kommuneid 
print(df.columns)

#print(df["Pasienter"].head(10))
# Bring into memory (make sure it's not too big!)



#df = df.compute()

print("\nAfter compute, dtypes:")
print(df.dtypes)

for col, dtype in df.dtypes.items():
    print(f"{col}: {dtype}")



print(df['age'])
# Optional: Convert age to ordered category

#df = df.compute()
category_order = [
    "0 <= x < 5", "5 <= x < 10", "10 <= x < 15", "15 <= x < 20",
    "20 <= x < 25", "25 <= x < 30", "30 <= x < 35", "35 <= x < 40",
    "40 <= x < 45", "45 <= x < 50", "50 <= x < 55", "55 <= x < 60",
    "60 <= x < 65", "65 <= x < 70", "70 <= x < 75", "75 <= x < 80",
    "80 <= x < 85", "85 <= x < 90", "90 <= x < 95", "95 <= x <= 100"
]
df['age_cat'] = pd.Categorical(df['age'],categories=category_order,ordered=True)
df['age_group'] = df['age_cat'].cat.codes

for col in ['Omsorg', 'Avdeling', 'Hastegrad', 'Kjønn','age_cat']:
    df[col] = df[col].astype('category')


print(df['age_group'])

df['EPISODESLUTTTID'] = df['EPISODESLUTTTID'].astype('str')
if 'Kommune' in df.columns:
    df['Kommune']=df['Kommune'].astype('category')
    df = df.drop(columns=['KOMMUNEID'])

#print(df['episode_end_datetime'])
df = df.sort_values(by=['PERSONID'])
df = df.dropna(subset=['PERSONID'])
df['id']=pd.factorize(df['PERSONID'])[0]+1
df = df.sort_values(by=['id'])

print(df['id'])
df = df.sort_values(by='id').reset_index(drop=True)



print("start writing to disk")
df.to_csv("data/rdaps.csv")
print("write parquet")
df.to_parquet("data/rdaps.parquet")
print("finished") 
    

