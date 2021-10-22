"""Model used for the prediction"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

def yle_model(userdf):
   
   # 1 LOAD AND INSPECT DATA
   yle_data = pd.read_csv('/home/chpatola/Desktop/Skola/DataSci/Project_work/yle_data.csv',
                          sep=';',
                          encoding="ISO-8859-1"
                          ) 

   # 2 CHOOSE COLUMNS WE WANT TO KEEP
   data = yle_data.loc[:,['id','ikä','sukupuoli','valittu','julkkis','puolue','Toimin tällä hetkellä kansanedustajana.','Koulutus','Kielitaito',
   'äidinkieli','Twitter-profiilin osoite:','Lapsia','Työnantaja','Ammattiasema','Käytän vaaleihin rahaa','Tärkein ulkopuolinen rahoituslähde','Vuositulot']]

   # 3 RESHAPE SPECIFIC DATA BEFORE NAN-REMOVAL

   #3.1 Merge similar Ammattiasema with each other
   data.Ammattiasema.value_counts()
   data.Ammattiasema.replace("asiantuntijatehtävä ", "expert/highly_ed",inplace = True)
   data.Ammattiasema.replace("toimihenkilö ", "expert/highly_ed" , inplace=True)
   data.Ammattiasema.replace("korkeakoulututkintoa vaativa tehtävä ", "expert/highly_ed",inplace = True)
   data.Ammattiasema.value_counts()

   #3.2 Merge similar Koulutus with each other
   data.Koulutus.value_counts()
   data.Koulutus.replace("ammattitutkinto ", "vocational or matric. exam",inplace = True)
   data.Koulutus.replace("ylioppilas ", "vocational or matric. exam" , inplace=True)
   data.Koulutus.value_counts()

   #3.3 Translate Column names to english
   def nyName (old,new):
      data.rename(columns={old: new}, inplace = True)

   nyName("ikä","age")
   nyName("sukupuoli","sex")
   nyName("valittu", "elected")
   nyName("julkkis", "celebrity")
   nyName("puolue", "party")
   nyName("äidinkieli", "mother_tongue")
   nyName("Lapsia", "children")
   nyName("Työnantaja", "employer")
   nyName("Ammattiasema", "work_status")
   nyName("Vuositulot", "yearly_income")
   nyName("Kielitaito", "languages")
   nyName("Käytän vaaleihin rahaa", "elect_budget")
   nyName("Koulutus", "education")
   nyName("Tärkein ulkopuolinen rahoituslähde", "ext_election_funding")
   nyName("Toimin tällä hetkellä kansanedustajana.", "currently_in_parliament")
   nyName("Twitter-profiilin osoite:", "twitter_account")
   data.columns

   #3.4 Create new column containing number of languages spoken
   data['languages_known']=data['languages'].str.count(' ')
   data.drop(columns=['languages'], inplace = True)

   #3.5 Create new column based on member in current prime minister party
   # (no NaN:s in this column) or not. 
   def label_race (row):
      if row['party'] == 'Kansallinen Kokoomus' :
         return 1
      return 0

   data['PM_party']=data.apply (lambda row: label_race(row), axis=1)
   data.drop(columns='party', inplace=True)

   # 4 INSPECTION AND HANDLING OF NAN
   data.isna().sum()
   data[data.age==0].count() # we have candidates with age 0
   dataage =data[data.age !=0]
   medianIkä = dataage.age.median()
   data.age.replace(0, medianIkä, inplace = True)
   data[data.age ==0] # No 0-years old left

   #4.1.2 Change NaN twitter to 0 och 1 instead of 1 and NaN
   data.twitter_account.fillna(0, inplace=True)

   #4.2 Erase rows with too many NaN:s (we want high quality answers)
   data.isna().sum()
   data.dropna(axis = 0, inplace = True, thresh= 15)

   #4.3 Handle NaN s for the rest of the columns
   #We impute the NaN:s with the most common value and create separate columns to hold track of 
   #rows where we imputed a NaN for a specific columnd

   #Check that no NA in to be y columns
   data.elected.isna().sum()

   #Copy data not to mess with the original one
   data_nonan = data.copy()
   my_imputer = SimpleImputer(strategy = "most_frequent")
   data_imp = pd.DataFrame(my_imputer.fit_transform(data_nonan))
   #Simple Imputer took away column names so we put them back
   data_imp.columns = data_nonan.columns  
   data_imp.isna().sum() # No more NaN:s in the DF!
   data_imp.dtypes#We need to fix the numeric data types back!
   data_imp[["id", "age","celebrity","currently_in_parliament","twitter_account",
   "languages_known","PM_party"]] = data_imp[["id", "age","celebrity",
   "currently_in_parliament","twitter_account","languages_known","PM_party"]].apply(pd.to_numeric)
   data_imp.dtypes

   # 5 SEPARATE INTO X AND y
   X = data_imp.drop(axis=1, columns=['elected','id'])
   X.columns
   y = data.elected

   data1 = X

   #6 RESHAPE DATA AFTER NAN-HANDLING: creating additional column, categorical values to numbers

   #6.1 Create new column with info on external financing or not

   def label_race2 (row):
      if row['ext_election_funding'] == 'ei ulkopuolista vaalirahoitusta' :
         return 0
      return 1   

   data1['external_election_funding']=data1.apply (lambda row: label_race2(row), axis=1)
   data1.drop(columns='ext_election_funding', inplace=True)

   #7 RESHAPE CATEGORICAL VALUES

   #7.1 Transform columns with ordered categories: vaalibudjetti, Vuositulot
   def fund_race (row):
      if row['elect_budget'] ==  "alle 1 000 euroa ":
         return 500
      if row['elect_budget'] ==  "1 000-5 000 euroa ":
         return 3000
      if row['elect_budget'] ==  "5 000-10 000 euroa ":
         return 7500
      if row['elect_budget'] ==  "10 000-20 000 euroa ":
         return 15000
      if row['elect_budget'] ==  "20 000-50 000 euroa ":
         return 35000
      return 90000

   def income_race (row):
      if row['yearly_income'] ==  "alle 20 000 euroa ":
         return 10000
      if row['yearly_income'] ==  "20 000-30 000 euroa ":
         return 25000
      if row['yearly_income'] ==  "30 000-50 000 euroa ":
         return 40000
      if row['yearly_income'] ==  "50 000-70 000 euroa ":
         return 60000
      if row['yearly_income'] ==  "70 000-100 000 euroa ":
         return 85000
      return 130000

   data1['elect_budget_new']=data1.apply (lambda row: fund_race(row), axis=1)
   data1['yearly_income_new']=data1.apply (lambda row: income_race(row), axis=1)
   data1.drop(columns=['elect_budget','yearly_income'], inplace=True)

   #7.2 Lable-encoder for the unordered categorical data
   # Make copy to avoid changing original data 
   label_X_train = data1.copy()

   # Apply label encoder to choosen columns
   label_encoder = LabelEncoder()
   categorical_feature_mask = data1.dtypes==object
   categorical_cols = data1.columns[categorical_feature_mask].tolist()

   for col in categorical_cols: 
      label_X_train[col] = label_encoder.fit_transform(data1[col])

   #*******This (above) is the way the input from the user will be like
   
   #7.3 Onehotencoder for the columns we put label encoder on 

   #Define columns to use it on
   categorical_cols_version2 =  categorical_cols.copy()
   categorical_cols_version2.remove('sex')
   categorical_cols_version2.remove('children')

   # Apply one-hot encoder to each column with categorical data
   OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
   OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(label_X_train[categorical_cols_version2]))

   # One-hot encoding removed index; put it back
   OH_cols_train.index = label_X_train.index

   # Remove categorical columns (will replace with one-hot encoding)
   num_X_train = label_X_train.drop(categorical_cols_version2, axis=1)

   # Add one-hot encoded columns to numerical features
   OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

   Final_X = OH_X_train

   #8  DIVIDE INTO TRAIN AND TEST
   train_X, val_X, train_y, val_y = train_test_split(Final_X, y, test_size = 0.3,random_state = 1,stratify =y)
 
   #9 DEFINE AND FIT MODEL
   #Transform values to correct shape for XgB model
   train_x_values = train_X.values
   val_x_values = val_X.values
   userdef_values = userdf.values

   xg_reg = xgb.XGBClassifier(subsample=0.2,min_samples_split = 75,
   max_features = 'sqrt',n_estimators=8)
   xg_reg.fit(train_x_values, train_y)

   #10 USE MODEL FOR PREDICTIONS

   #10.1 Predict on val_X
   predictions = xg_reg.predict(val_x_values)

   #10.2 Predict on user data
   userd_newformat= pd.DataFrame(xg_reg.predict_proba(userdef_values))
   percent_predictions_user =np.around((userd_newformat.iloc[0,1])*100,decimals = 3)#Gives prob
   # for the row to belong to class 1 = seat in parliament

   classif_report = classification_report(val_y, predictions)
   '''
   print(accuracy_score(val_y, predictions))

   '''
   
   
   return percent_predictions_user, classif_report
   

#Test function with this data

'''
mylist0 = [32,0,0,0,0,0,0,0,0,0,0,0,0,500,10000]
mylist1 = [32,1,1,1,1,1,1,1,1,1,1,1,1,500,10000]
mylist2 = [32,1,1,1,2,2,1,1,2,2,2,1,1,500,10000]
mylist3 = [32,1,1,1,3,0,1,1,3,3,0,1,1,500,10000]
mylist4 = [32,1,1,1,1,0,1,1,2,4,0,1,1,500,10000]
mylist5 = [32,1,1,1,1,0,1,1,0,5,0,1,1,500,10000]
mylist6 = [32,1,1,1,1,0,1,1,1,6,0,1,1,500,10000]
mylist7 = [32,1,1,1,1,0,1,1,1,7,0,1,1,500,10000]
mylist8 = [32,1,1,1,1,0,1,1,1,8,0,1,1,500,10000]

mydataf = pd.DataFrame([mylist0, mylist1,mylist2,mylist3,mylist4,mylist5,mylist6,mylist7,mylist8],columns =['age',
        'sex', 'celebrity', 'currently_in_parliament', 'education',
       'mother_tongue', 'twitter_account', 'children', 'employer',
       'work_status', 'languages_known', 'PM_party',
       'external_election_funding', 'elect_budget_new ', 'yearly_income_new'])

mydataf.shape#15 columns    
crit1 = mydataf.dtypes!=object
cat2 = mydataf.columns[crit1].tolist()
cat3 = cat2[4:10]
cat3.remove('children')
cat3.remove('twitter_account')
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train_user = pd.DataFrame(OH_encoder.fit_transform(mydataf[cat3]))

OH_cols_train_user.index = mydataf.index
num_X_train2 = mydataf.drop(cat3, axis=1)

OH_X_user = pd.concat([num_X_train2, OH_cols_train_user], axis=1)
OH_X_user.shape#31 columns


print('sending data to yle_model')
chance, statistics = yle_model(OH_X_user)

print("Probability to get elected is")
print(chance)
print("Model info\n")
print(statistics)
'''









