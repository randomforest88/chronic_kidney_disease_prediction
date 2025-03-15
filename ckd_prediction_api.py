from pyscript import document

r_identities = ["#ada_d","#ada_r","#ext_d","#dec_d","#storeadj"]
output_div = []
for l in r_identities:
  output_div.append(document.querySelector(l))

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

userinput = []
user2input = []
def submitbtn(event):
  for i in range(3,8):
    if (document.querySelector("#d"+str(i)).checked == True):
      userinput.append(1)
    else:
      userinput.append(0)
    user2input.append(document.querySelector("#d"+str(i)).checked)

  data = {
    "age": [document.querySelector("#ageinput").value],
    "blood_pressure": [document.querySelector("#bpinput").value],
    "hypertension": [userinput[0]],
    "diabetes_mellitus": [userinput[1]],
    "coronary_artery_disease": [userinput[2]],
    "appetite": [document.querySelector("#storeappetite").value],
    "pedal_edema": [userinput[3]],
    "anaemia": [userinput[4]]
  }

  input_data = pd.DataFrame(data)

  df= pd.read_csv('kidney_disease.csv')
  df.drop('id', axis = 1, inplace = True)
  df.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
              'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
              'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
              'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'pedal_edema',
              'anaemia', 'class']

  df['packed_cell_volume'] = pd.to_numeric(df['packed_cell_volume'], errors='coerce')
  df['white_blood_cell_count'] = pd.to_numeric(df['white_blood_cell_count'], errors='coerce')
  df['red_blood_cell_count'] = pd.to_numeric(df['red_blood_cell_count'], errors='coerce')
  cat_cols = [col for col in df.columns if df[col].dtype == 'object']
  num_cols = [col for col in df.columns if df[col].dtype != 'object']

  df['diabetes_mellitus'].replace(to_replace = {'\tno':'no','\tyes':'yes',' yes':'yes'},inplace=True)
  df['coronary_artery_disease'] = df['coronary_artery_disease'].replace(to_replace = '\tno', value='no')
  df['class'] = df['class'].map({'ckd': "yes", 'notckd': "no", 'ckd\t':"yes"})
  cols = ['diabetes_mellitus', 'coronary_artery_disease', 'class']
  df.isna().sum().sort_values(ascending = False)

  def random_value_imputation(feature):
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    random_sample.index = df[df[feature].isnull()].index
    df.loc[df[feature].isnull(), feature] = random_sample
    
  def impute_mode(feature):
    mode = df[feature].mode()[0]
    df[feature] = df[feature].fillna(mode)

  for col in num_cols:
    random_value_imputation(col)

  random_value_imputation('red_blood_cells')
  random_value_imputation('pus_cell')

  for col in cat_cols:
    impute_mode(col)


  le = LabelEncoder()

  for col in cat_cols:
    df[col] = le.fit_transform(df[col])

  #scaling
  ind_col = [col for col in df.columns if col != 'class']
  dep_col = 'class'

  X = df[ind_col]
  y = df[dep_col]

  X.drop(['specific_gravity','albumin','sugar','red_blood_cells','pus_cell','pus_cell_clumps','bacteria','blood_glucose_random','blood_urea','serum_creatinine','sodium','potassium','haemoglobin','packed_cell_volume','white_blood_cell_count','red_blood_cell_count'], axis = 'columns', inplace = True)

  scaler=StandardScaler()
  X['age']=scaler.fit_transform(X[['age']])
  input_data['age']=scaler.transform(input_data[['age']])


  #Ada Boost Model
  ada = AdaBoostClassifier(n_estimators=100, random_state = 42)
  ada.fit(X, y)

  result = [ada.predict(input_data),ada.predict_proba(input_data)]
  riskindex = result[1].reshape(-1)[1]

  if (riskindex < 0.5):
    output_div[4].value = "0"
  elif (riskindex <=0.6):
    output_div[4].value = "1"
  elif (riskindex <=0.8):
    output_div[4].value = "2"
  elif (riskindex <=0.9):
    output_div[4].value = "3"
  else:
    output_div[4].value = "4"

  output_div[0].innerHTML = "Diagnosis by model: "+str(result[0] == [1])
  output_div[1].innerHTML = "Risk: "+str(round(riskindex*100,2))+"%"


  #Extra trees model
  extc = ExtraTreesClassifier()
  extc.fit(X, y)

  result = [extc.predict(input_data),extc.predict_proba(input_data)]
  riskindex = result[1].reshape(-1)[1]

  output_div[2].innerHTML = "Diagnosis by model: "+str(result[0] == [1])

  #Decision Tree Model
  dtc = DecisionTreeClassifier(random_state=42)
  dtc.fit(X, y)

  result = [dtc.predict(input_data),dtc.predict_proba(input_data)]
  riskindex = result[1].reshape(-1)[1]

  output_div[3].innerHTML = "Diagnosis by model: "+str(result[0] == [1])
