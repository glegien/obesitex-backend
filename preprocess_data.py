import pandas as pd
# set data path
#data_path = os.path.join(PROJECT_ROOT, 'data', 'BIONINJACONTEST', 'Otylosc')

Pheno_SEX_AGE_ZAKODOWANE_xlsx = os.path.join(data_path, 'Pheno_SEX_AGE_ZAKODOWANE.xlsx') 
df = pd.read_excel(Pheno_SEX_AGE_ZAKODOWANE_xlsx)
Pheno_SEX_AGE_ZAKODOWANE_xlsx_df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

Pheno_log_BMI_zakodowany_xlsx = os.path.join(data_path, 'Pheno_log_BMI_zakodowany.xlsx') 
df = pd.read_excel(Pheno_log_BMI_zakodowany_xlsx)
Pheno_log_BMI_zakodowany_xlsx_df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

PHENO_zakodowany_txt = os.path.join(data_path, 'PHENO_zakodowany.txt') 
PHENO_zakodowany_txt_df = pd.read_csv(PHENO_zakodowany_txt,sep='\t')#, names=['FID',

df_1 = Pheno_log_BMI_zakodowany_xlsx_df.drop(columns=['FID'])
df_2 = PHENO_zakodowany_txt_df.drop(columns=['FID'])
df_3 = Pheno_SEX_AGE_ZAKODOWANE_xlsx_df.drop(columns=['FID'])

df = pd.merge(df_1, df_2, on='IID')
df = pd.merge(df, df_3.loc[:, df_3.columns != 'SEX'], on='IID')
# df[(df['nor_over_M'] == 1) & (df['nor_ovob_M'] == 0) ]
df.loc[:, df.columns != 'IID'] = df.loc[:, df.columns != 'IID'].replace([1,2], [0,1])
def fat_state_female(row):
   if row['nor_obe_F'] == 1 :
      return 'obesity'
   elif row['nor_ovob_F'] == 1 :
      return 'obesity'
   elif row['nor_over_F'] == 1 :
      return 'overweight'
   elif ((row['nor_ovob_F'] == 0) and (row['nor_obe_F'] == 0)) and (row['nor_over_F'] == 0):  
      return 'normal'
   else:
      return 'other_sex'
    
def fat_state_male(row):
   if row['nor_obe_M'] == 1 :
      return 'obesity'
   elif row['nor_ovob_M'] == 1 :
      return 'obesity'
   elif row['nor_over_M'] == 1 :
      return 'overweight'
   elif ((row['nor_ovob_M'] == 0) and (row['nor_obe_M'] == 0)) and (row['nor_over_M'] == 0):  
      return 'normal'
   else:
      return 'other_sex' 
df_before_cat = df.copy()

df['female_state'] = df.apply(lambda row: fat_state_female(row), axis=1)
df['malee_state'] = df.apply(lambda row: fat_state_male(row), axis=1)
df = df.drop(columns=[ 
         'nor_obe_all',
         'nor_ovob_all',
         'nor_over_all',
         'nor_obe_F',
         'nor_ovob_F',
         'nor_over_F',
         'nor_obe_M',
         'nor_ovob_M',
         'nor_over_M'])
df = df.replace([-9., -9], [pd.NaT, pd.NaT])
df = df.dropna(how='any')

def load_genom():
    header_file = open(os.path.join(data_path, 'BioNinjaHack_obesity.map'))
    headers = [line.split()[1] for line in header_file]
    headers = ["IID"] + headers

    res = [headers]
                       
    genome_file = open(os.path.join(data_path, "BioNinjaHack_obesity.ped"))
    for line in genome_file:
        text = line.split()
        id_ = text[1]
        record = [id_]
        genome = text[6:]
        for x in range(0, len(genome), 2):
            record.append(genome[x] + genome[x + 1])
        res.append(record)
    return res

lol = load_genom()
headers = lol.pop(0) # gives the headers as list and leaves data
df_genome = pd.DataFrame(lol, columns=headers)
df_genome['IID'] = df_genome['IID'].astype(int)
df_all = pd.merge(df, df_genome, on='IID')
df_all['log_BMI'] = df_all['log_BMI'].astype(float)
df_all['AGE'] = df_all['AGE'].astype(int)
df_all = df_all.loc[:,~df_all.columns.duplicated()]
index_to_drop = df_all[(df_all['malee_state'] == 'other_sex')
                       & (df_all['female_state'] == 'other_sex')].index
df_all.drop(index_to_drop, inplace=True)
df_all.to_csv(os.path.join(data_path, 'all_data.csv'))

df_female = df_all[df_all["SEX"] == 1].drop(columns=['SEX', 'malee_state']).rename(columns={'female_state':'is_obesity'}) 
index_to_drop = df_female[(df_female['is_obesity'] == 'other_sex')].index
df_female.drop(index_to_drop, inplace=True)
df_female['is_obesity'].replace({'normal': 0, 'obesity': 1}, inplace=True)
df_female.to_csv(os.path.join(data_path, 'female_data.csv'))

df_male = df_all[df_all["SEX"] == 0].drop(columns=['SEX', 'female_state']).rename(columns={'malee_state':'is_obesity'}) 
index_to_drop = df_male[(df_male['is_obesity'] == 'other_sex')].index
df_male.drop(index_to_drop, inplace=True)
df_male['is_obesity'].replace({'normal': 0, 'obesity': 1}, inplace=True)
df_male.to_csv(os.path.join(data_path, 'male_data.csv'))
