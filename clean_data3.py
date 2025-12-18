import pandas as pd
import sys
import dask.dataframe as dd
import simple_icd_10 as icd
import icd10
import simple_icd_10_cm as cm
import re

df = dd.read_parquet("data/rdaps.parquet")
df = df.dropna(subset=['PERSONID'])
length = df.shape[0].compute()
df = df.compute()
#length = df.shape[0].compute()
print(f"Number of rows in filtered data: {length}")

df = df.rename(columns={'Kjønn':'Sex'})
df = df.rename(columns={'ÅrEpisode':'YearEpisode'})


print(df['Bidiagnose'].notna())
non_empty = df[df['Bidiagnose'].notna() & (df['Bidiagnose'].str.strip() != "")  & (df['Bidiagnose'].str.strip() != "-")]
print(non_empty['Bidiagnose'])

def remove_aa_from_start(txt):
    if txt[0] == 'å' and txt[1] == 'å' and txt[3] == 'å':
        txt = txt[3:]
    elif txt[0] == 'å' and txt[1] == 'å':
        txt = txt[2:]
    elif txt[0] == 'å':
        txt = txt[1:]

    return txt

def is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# Pre-compile a regex for a valid ICD-10 base (letter + 2 digits)
_VALID_BASE = re.compile(r'^[A-Z][0-9]{2}$')

def split_bidiagnosis(text):
    if len(text) > 4:
        list = text.split(" ")
        code = list[0]
        return(code)





df['bidiagnosiscode'] = df['Bidiagnose'].apply(split_bidiagnosis)



#non_empty = df[df['bidiagnosiscode'].notna() & (df['bidiagnosiscode'].str.strip() != "")  & (df['bidiagnosiscode'].str.strip() != "-")]
#print(non_empty['bidiagnosiscode'])

def get_icd10_group_number(icd10_code):
    """
    Map an ICD-10 code to its corresponding block (group) code.
    Returns something like "A00-A09" rather than the full description.
    """
    if pd.isna(icd10_code):
        return "Unknown"

    icd10_code = str(icd10_code).strip().upper()

    
    

    # Handle empty or invalid codes
    if not icd10_code or icd10_code == 'NAN' or icd10_code == "-" or is_int(icd10_code):
        return "Unknown"


    if pd.isna(icd10_code):
        return "Missing diagninosis"
    
    icd10_code = str(icd10_code).strip().upper()

    # Handle empty or invalid codes
    if not icd10_code or icd10_code == 'NAN' or icd10_code == "-":
        #if icd10_code != None:
        #    print(f'{icd10_code} missig')
        
        return "Misssing diagnosis"

    #if icd10_code == "V2S9" | icd10_code == "Y4N" | icd10_code == "Y1NX":
    #    return "V01–Y98"
    
    if len(icd10_code) > 4:
       icd10_code =icd10_code[:4] 

    try:
        icd10_code = icd.add_dot(icd10_code)
    except:
        pass

    
    if icd10_code and _VALID_BASE.match(icd10_code[:3]):
        try:
            code=icd.get_ancestors(icd10_code)
            code=code[-2]
            return code
        except:
            pass  
        

    try:
        if len(icd10_code) > 3: 
                tmp = icd10_code[:3]
                code = icd.get_ancestors(tmp)
                code = code[-2]
                return code
    except:
        pass

    if len(icd10_code) > 1:
        if icd10_code == "B59":
            return "A00–B99"

    if len(icd10_code) > 1:
        if icd10_code[0] == 'T':
            return "Injury, poisoning and certain other consequences of external causes"

    if len(icd10_code) > 1:
        if icd10_code[0] == 'X':
            return "Self Injury"

   
   
    if len(icd10_code) > 1:
        if icd10_code[0] == 'U':
            return "U00–U99 included Covid"

    if is_int(icd10_code):
        return "F00–F99"

    if is_float(icd10_code):
        return "F00–F99"

    if len(icd10_code) > 1:
        #print(type(icd10_code))
        #print(icd10_code[0])
        if icd10_code[0] == "V" or icd10_code[0] == "W" or icd10_code[0] == "Y":
                return "External causes of morbidity and mortality"


    if  icd10_code != None:
        return icd10_code
    else:
        return "Unknown diagnosis"




def get_icd10_group_name(icd10_code):
    """
    Map an ICD-10 code to its corresponding chapter/group name.
    """
    if pd.isna(icd10_code):
        return "Error diagnosis missing from rdap"
    
    icd10_code = str(icd10_code).strip().upper()

    # Handle empty or invalid codes
    if not icd10_code or icd10_code == 'NAN' or icd10_code == "-":
        #if icd10_code != None:
        #    print(f'{icd10_code} missig')
        
        return "Error diagnosis missing from rdap"

    #if icd10_code == "V2S9" | icd10_code == "Y4N" | icd10_code == "Y1NX":
    #    return "External causes of morbidity and mortality"

    if len(icd10_code) > 4:
       icd10_code =icd10_code[:4] 

    try:
        icd10_code = icd.add_dot(icd10_code)
    except:
        pass

    
    if icd10_code and _VALID_BASE.match(icd10_code[:3]):
        try:
            code=icd.get_ancestors(icd10_code)
            code=code[-1]
            out=icd.get_description(code)
            return out
        except:
            pass  
        

    try:
        if len(icd10_code) > 3: 
                tmp = icd10_code[:3]
                code = icd.get_ancestors(tmp)
                code = code[-1]
                out=icd.get_description(code)
                return out
    except:
        pass

    if len(icd10_code) > 1:
        if icd10_code == "B59":
            return "Certain infectious and parasitic diseases"
    
    if len(icd10_code) > 1:
        if icd10_code[0] == 'U':
            return "U00–U99 included Covid"

    if len(icd10_code) > 1:
        if icd10_code[0] == 'T':
            return "Injury, poisoning and certain other consequences of external causes"

    if len(icd10_code) > 1:
        if icd10_code[0] == 'X':
            return "Self Injury"



    if ',' in icd10_code:
        dsm = icd10_code.split(',')[0]
        if is_int(dsm) | is_float(dsm):
            return "Mental and behavioural disorders"

    if is_int(icd10_code):
        return "Mental and behavioural disorders"

    if is_float(icd10_code):
        return "Mental and behavioural disorders"

    if len(icd10_code) > 1:
        #print(type(icd10_code))
        #print(icd10_code[0])
        if icd10_code[0] == "V" or icd10_code[0] == "W" or icd10_code[0] == "Y":
                return "External causes of morbidity and mortality"        

    if  icd10_code != None:
        return icd10_code
    else:
        return "Unknown diagnosis"


df['EPISODESLUTTTID']=df['EPISODESLUTTTID'].astype('string')
df['EPISODESTARTTID'] = df['EPISODESTARTTID'].astype('string')

df['EPISODESLUTTTID']=df['EPISODESLUTTTID'].str.strip()
df['EPISODESTARTTID'] = df['EPISODESTARTTID'].str.strip()

df['EPISODESLUTTTID'] = df['EPISODESLUTTTID'].str.replace(r'[^\x00-\x7F]+', '', regex=True) 
df['EPISODESLUTTTID'] = df['EPISODESLUTTTID'].str.replace(r'[^\x00-\x7F]+', '', regex=True)


df['episode_start_datetime'] =pd.to_datetime(df['EPISODESTARTTID'],format='%Y-%m-%d %H:%M:%S', errors='coerce')

                                         
                                             
                                             #,format='%Y-%m-%d %H:%M.%S', errors='coerce')
df['episode_end_datetime'] =pd.to_datetime(df['EPISODESLUTTTID'],format='%Y-%m-%d %H:%M:%S', errors='coerce')

def is_icd10_code(token):
    """
    Check if token looks like an ICD-10 code: e.g., A00, B20.1, C50.91
    """
    return bool(re.match(r'^[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$', token.upper()))


def secondary_all_diagnsosis(text):
    diagnosis = set() 
    listtxt = text.split(" ")
    #print(listtxt)
    for i in listtxt:
        if len(i) > 5:
            continue
        group = get_icd10_group_name(i)
        clean_group = group.strip().title()  # Normalize format
        if len(clean_group) >5:
            diagnosis.add(clean_group)        
    out = ",".join(diagnosis)
    #print(out)
    return out
df['bidiagnosiscode']=df['Bidiagnose'].apply(secondary_all_diagnsosis)




print(df[['id', 'episode_start_datetime', 'episode_end_datetime']])
print("before remove and merge of dupliacte rows.")
print(df.columns)
df = (
    df.groupby(['id', 'episode_start_datetime'], as_index=False)
    .agg({
        **{col: 'first' for col in df.columns if col not in ['bidiagnosiscode']},
        'bidiagnosiscode': lambda x: ', '.join(sorted(set(x.dropna())))
    })
)

print(df['bidiagnosiscode'])
print(type(df))
print("test2")

print(df.columns)

for col in ['Omsorg', 'Avdeling', 'Hastegrad', 'Sex','age_cat']:
    df[col] = df[col].astype('category')



#df['icd10_from_main']=df['HovedDiagnose'].apply(get_icd10_group_name)





df['ICD10_code_diagnosis']=df['HovedDiagnosekode'].apply(get_icd10_group_name)
df['ICD10_code_diagnosis_group']=df['HovedDiagnosekode'].apply(get_icd10_group_number)


non_empty = df[df['ICD10_code_diagnosis'].notna() & (df['ICD10_code_diagnosis'].str.strip() != "Error diagnosis missing from rdap")  & (df['ICD10_code_diagnosis'].str.strip() != "-")]
print(non_empty['ICD10_code_diagnosis'])
#print(df['ICD10_code_diagnosis'])

#imask=(df['Avdeling] ="BUP-ST" & df['HovedDiagnosekode'] != "-")

#df_sorted.loc[imask,['ICD10_code_diagnosis_group']] = "Mental and behavioural disorders"


non_empty = df[df['bidiagnosiscode'].notna() & (df['bidiagnosiscode'].str.strip() != "Error diagnosis missing from rdap")  & (df['bidiagnosiscode'].str.strip() != "-")]
print(non_empty['bidiagnosiscode'])

df['Lokasjon'] = df['Lokasjon'].apply(remove_aa_from_start)
df['Avdeling'] = df['Avdeling'].apply(remove_aa_from_start)

for col in df.columns:
    print(f"{col}: {df[col].dtype}")









df['treatment_time'] = (df['episode_end_datetime'] - df['episode_start_datetime']).fillna(pd.Timedelta(seconds=0))
df['duration_hours'] = df['treatment_time'].dt.total_seconds() / 3600
print(df['duration_hours'])
print(f"Total memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

df_sorted = df.sort_values(by=['id','episode_start_datetime'])
df_sorted = df_sorted.reset_index(drop=True)
# Set index for better partitioning (expensive but scalable)
#df_sorted = df.set_index('id').map_partitions(
 #   lambda df: df.sort_values(by='episode_start_datetime')
#)

#del df
df_sorted['readmission'] = 0
df_sorted['prev_episode_start'] = df_sorted['episode_start_datetime'].shift()
df_sorted['prev_id'] = df_sorted['id'].shift()
df_sorted['prev_readmission'] = df_sorted['readmission'].shift()
df_sorted['prev_treatment_time'] = df_sorted['treatment_time'].shift()
df_sorted['pre_HovedDiagnosekode'] = df_sorted['HovedDiagnosekode'].shift()
df_sorted["visits"] = df_sorted.groupby("id").cumcount() + 1

#for index, row in df_sorted.iterrows():
#    if (df_sorted['prev_id'][index] == df_sorted['id'][index] and df_sorted['episode_start_datetime'][index] - df_sorted['prev_episode_start'][index] < pd.Timedelta(days=30)):
#        df_sorted.loc[index,'readmission']= 1
#        print(row)

mask= (df_sorted['prev_id'] == df_sorted['id']) & ( (df_sorted['episode_start_datetime'] - df_sorted['prev_episode_start'] < pd.Timedelta(days=30)) | (df_sorted['episode_start_datetime'] - (df_sorted['prev_treatment_time'] + df_sorted['prev_episode_start']  )< pd.Timedelta(days=30)) )
#print(mask)
#readmissions_within_30_days =mask
#print(readmissions_within_30_days)
#df_sorted['readmission'] = mask

df_sorted.loc[mask,['readmission']] = 1

print(df_sorted['readmission'])
readadimmsion = df_sorted['readmission']
print("write csv")
df_sorted.to_csv("data/cleaned_data.csv.gz",index=False)
#df_sorted.to_parquet("data/cleaned_data.parque",engine="pyarrow")
print("finished writing csv")
y_df = pd.DataFrame(readadimmsion)

print(df_sorted.columns)

#y_df. = ['label']

y_df = y_df.rename(columns={'readmission': 'label'})


x_df =df_sorted.drop(['readmission'],axis=1)
x_df =x_df.drop(["HovedDiagnosekode"],axis=1)
x_df =x_df.drop(["HovedDiagnose"],axis=1)
#x_df =x_df.drop(["visits"],axis=1)


x_df = x_df.drop(columns=['episode_start_datetime','episode_end_datetime','YearEpisode','Lokasjon','ICD10_code_diagnosis_group','prev_treatment_time','treatment_time'])

if "icd10_from_main" in x_df.columns:
    x_df = x_df.drop(columns=["icd10_from_main"])


print(x_df.columns)
x_df = x_df.drop(columns=['MndEpisode3','prev_episode_start','prev_episode_start','age','prev_id','age_group','EPISODESTARTTID','prev_episode_start','prev_readmission','age','EPISODESLUTTTID'])
x_df = x_df.drop(columns=['PERSONID'])
x_df = x_df.drop(columns=['pre_HovedDiagnosekode'])




if 'Kommune' in x_df.columns:
        x_df=x_df.drop(columns=['Kommune'])




print("write_xgo_files")
x_df.to_parquet('data/x_df.parque',engine="pyarrow")
y_df.to_parquet('data/y_df.parque',engine="pyarrow")

print("now make file without outpatient clinic")


mask = x_df['Omsorg'] != 'Poliklinisk omsorg'

x_no_out_patient = x_df[mask]
y_no_out_patient = y_df[mask]

x_no_out_patient.to_parquet('data/x_no_out_patient.parque',engine="pyarrow")
y_no_out_patient.to_parquet('data/y_no_out_patient.parque',engine="pyarrow")


print(f' number of indiviuals {len(x_no_out_patient)}')
print(f' numeber of readmited {y_no_out_patient['label'].sum()}')
print(f'number of not readmitted {len(x_no_out_patient) -y_no_out_patient['label'].sum()}')






