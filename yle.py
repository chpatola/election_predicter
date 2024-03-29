import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt


# 0 PREPROCESSING IN EXCEL-copy of original csv file

# 0.1 Insert column with names of elected, find doublets  and create new column based on result
# 0.2 Find names of celebrities (names found in articles online) and create new column based on this
# 0.3 Inspect column values for "Ammattiasema" for empty values in "Työnantaja" and fill in info if it can be deducted
# 0.4 Inspect values for *Perheesen kuulu" and deduct info for column *Lapsia'
# 0.5 Inspect values for "Ammatti" and deduct info for column "Ammattiasema'
# 0.6 Inspect empty values in "Sukupuoli" and fill in the info based in the name
# 0.7 Change all cells with value in Twitter to 1
# 0.8 Erase sensitive data

# 1 LOAD AND INSPECT DATA
yle_data = pd.read_csv(
    '/home/chpatola/Desktop/Skola/DataSci/Project_work/yle_data.csv', sep=';', encoding="ISO-8859-1")
headsyle = yle_data.head()
print(headsyle)
print('success yle')

yle_data.columns

# 2 CHOOSE COLUMNS WE WANT TO KEEP
data = yle_data.loc[:, ['id', 'ikä', 'sukupuoli', 'valittu', 'julkkis', 'puolue', 'Toimin tällä hetkellä kansanedustajana.', 'Koulutus', 'Kielitaito',
                        'äidinkieli', 'Twitter-profiilin osoite:', 'Lapsia', 'Työnantaja', 'Ammattiasema', 'Käytän vaaleihin rahaa', 'Tärkein ulkopuolinen rahoituslähde', 'Vuositulot']]
data.shape
data.dtypes

# 3 RESHAPE SPECIFIC DATA BEFORE NAN-REMOVAL

# 3.1 Merge similar Ammattiasema with each other
data.Ammattiasema.value_counts()
data.Ammattiasema.replace("asiantuntijatehtävä ",
                          "expert/highly_ed", inplace=True)
data.Ammattiasema.replace("toimihenkilö ", "expert/highly_ed", inplace=True)
data.Ammattiasema.replace(
    "korkeakoulututkintoa vaativa tehtävä ", "expert/highly_ed", inplace=True)
data.Ammattiasema.value_counts()

data.columns

# 3.2 Merge similar Koulutus with each other
data.Koulutus.value_counts()
data.Koulutus.replace("ammattitutkinto ",
                      "vocational or matric. exam", inplace=True)
data.Koulutus.replace(
    "ylioppilas ", "vocational or matric. exam", inplace=True)
data.Koulutus.value_counts()

# 3.3 Translate Column names to english


def nyName(old, new):
    data.rename(columns={old: new}, inplace=True)


nyName("ikä", "age")
nyName("sukupuoli", "sex")
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

# 3.4 Create new column containing number of languages spoken
data['languages_known'] = data['languages'].str.count(' ')
data.drop(columns=['languages'], inplace=True)
data.columns
data.head(10)
data.languages_known.describe()

# 3.5 Create new column based on member in current prime minister party
# (no NaN:s in this column) or not.


def label_race(row):
    if row['party'] == 'Kansallinen Kokoomus':
        return 1
    return 0


data['PM_party'] = data.apply(lambda row: label_race(row), axis=1)

data.head(3)
data.drop(columns='party', inplace=True)
data.columns


# 4 INSPECTION AND HANDLING OF NAN
NrNas = data.isna().sum()
print(NrNas)

# 4.1 Unify definition of NaN

# 4.1.1 Age 0 can be seen as NaN, we fill it with median
data[data.age == 0].count()  # we have 31 with age 0
dataage = data[data.age != 0]
medianIkä = dataage.age.median()
data.age.replace(0, medianIkä, inplace=True)
data[data.age == 0]  # No 0-years old left

# 4.1.2 Change NaN twitter to 0 och 1 instead of 1 and NaN
data.twitter_account.fillna(0, inplace=True)
data.shape  # 15 columns
data.columns

# 4.2 Erase rows with too many NaN:s
data.isna().sum()
# We have 7/17 columns without NaN. Considering quality of each observations, we allow 4 nans
data.dropna(axis=0, inplace=True, thresh=13)

data.shape  # Now we have 1799 observations
data.isna().sum()
data.dtypes

# 4.3 Handle NaN s for the rest of the columns
# We impute the NaN:s with the most common value and create separate columns to hold track of
# rows where we imputed a NaN for a specific columnd

# Copy data not to mess with the original one
data_nonan = data.copy()

# Select columns containing missing values
cols_with_missing = [col for col in data_nonan.columns
                     if data_nonan[col].isnull().any()]

# Make new columns with info on imputation
for col in cols_with_missing:
    data_nonan[col + '_was_missing'] = data_nonan[col].isnull()

my_imputer = SimpleImputer(strategy="most_frequent")
data_imp = pd.DataFrame(my_imputer.fit_transform(data_nonan))
# Simple Imputer took away column names so we put them back
data_imp.columns = data_nonan.columns
data_imp.columns
data_imp.shape  # 22 columns

data_imp.isna().sum()  # No more NaN:s in the DF!
data_imp.dtypes  # We need to fix the int data types back!
data_imp[["id", "age", "elected", "celebrity", "currently_in_parliament", "twitter_account",
          "languages_known", "PM_party"]] = data_imp[["id", "age", "elected", "celebrity",
                                                      "currently_in_parliament", "twitter_account", "languages_known", "PM_party"]].apply(pd.to_numeric)
data_imp.dtypes

data1 = data_imp
data1.columns

# 5. RESHAPE DATA AFTER NAN-HANDLING: creating additional column, categorical values to numbers

# 5.1 Create new column with info on external financing or not


def label_race2(row):
    if row['ext_election_funding'] == 'ei ulkopuolista vaalirahoitusta':
        return 0
    return 1


data1['external_election_funding'] = data1.apply(
    lambda row: label_race2(row), axis=1)
data1.head(3)
data1.drop(columns='ext_election_funding', inplace=True)
data1.columns
data1.shape  # 26 columns

print(data1.dtypes)

# 5.1.1 Write to csv file for visualization in Power BI
# data1.to_csv("yle_processed.csv")
'''
#Create visualization for Power BI
electeds = data1.loc[data1.elected ==0]
no_elected = data1.loc[data1.elected ==1]
electeds_age_lang_ele = electeds.loc[:,['age','languages_known']]
electeds_age_lang_ele.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

no_elected_age_lang_ele = no_elected.loc[:,['age','languages_known']]
no_elected_age_lang_ele.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
'''

# Look at all categorical data
'''
#Lista på ej numeriska kolumner:
s = (data1.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)

l = ()
object_col_to_dummis = object_cols.copy()
object_col_to_dummis.remove('elect_budget')
object_col_to_dummis.remove('yearly_income')
print(object_col_to_dummis)
'''
# 5.2 Transform categorical, non-ordinal columns to dummies
data2 = pd.get_dummies(data1, prefix=['sex', 'education', 'mother_tongue', 'children', 'employer', 'work_status', 'education_was_missing',
                                      'mother_tongue_was_missing', 'children_was_missing', 'employer_was_missing', 'work_status_was_missing', 'elect_budget_was_missing',
                                      'ext_election_funding_was_missing', 'languages_known_was_missing', 'yearly_income_was_missing'],
                       columns=['sex', 'education', 'mother_tongue', 'children', 'employer', 'work_status', 'education_was_missing',
                                'mother_tongue_was_missing', 'children_was_missing', 'employer_was_missing', 'work_status_was_missing', 'elect_budget_was_missing',
                                'ext_election_funding_was_missing', 'languages_known_was_missing', 'yearly_income_was_missing'])
data2.shape
data2.columns

# 5.3 Change non-ASCII characters in headlines for lgb-model


def nyName1(old, new):
    data2.rename(columns={old: new}, inplace=True)


nyName1("children_kyllä ", "children_yes")
nyName1("employer_ei työelämässä", "employer_not_working")
nyName1("work_status_eläkeläinen ", "work_status_pensioner")
nyName1("work_status_maanviljelijä ", "work_status_farmer")
nyName1("work_status_työntekijä ", "work_status_worker")
nyName1("work_status_yrittäjä ", "work_status_entrepreneur")
data2.columns


# 5.4 Transform multi categorical, ordinal columns
def fund_race(row):
    if row['elect_budget'] == "alle 1 000 euroa ":
        return 500
    if row['elect_budget'] == "1 000-5 000 euroa ":
        return 3000
    if row['elect_budget'] == "5 000-10 000 euroa ":
        return 7500
    if row['elect_budget'] == "10 000-20 000 euroa ":
        return 15000
    if row['elect_budget'] == "20 000-50 000 euroa ":
        return 35000
    return 90000


def income_race(row):
    if row['yearly_income'] == "alle 20 000 euroa ":
        return 10000
    if row['yearly_income'] == "20 000-30 000 euroa ":
        return 25000
    if row['yearly_income'] == "30 000-50 000 euroa ":
        return 40000
    if row['yearly_income'] == "50 000-70 000 euroa ":
        return 60000
    if row['yearly_income'] == "70 000-100 000 euroa ":
        return 85000
    return 130000


data2['elect_budget_new'] = data2.apply(lambda row: fund_race(row), axis=1)
data2['yearly_income_new'] = data2.apply(lambda row: income_race(row), axis=1)
data2.drop(columns=['elect_budget', 'yearly_income'], inplace=True)
data2.columns
data2.describe()
data2.shape

# 6 DIVIDE INTO X AND Y AND TRAINING AND TEST-DATA

# 6.0 Check which X varibels have the strongest correlation to y
corres = data2.corr(method='spearman')
corres.elected.sort_values(ascending=False)  # interesting facts here...

# 6.1 Divide into x and y and drop identification columnd
X = data2.drop(axis=1, columns=['elected', 'id', 'PM_party'])
X.dtypes
X.shape
X.head(2)
y = data2.elected
y.head(3)

# Find X columns with high correlations
corr_values = X.corr(method='spearman').abs()
s = corr_values.unstack()
so = s.sort_values(ascending=False)
print(so[60:90])  # Almost all have to do with dummy yes/no-columns

# 6.2 Divide into train and test
seed = 0
scoring = 'accuracy'

train_X, val_X, train_y, val_y = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y)
train_X.dtypes

train_X.to_csv("yle_ready_x.csv")

# 7 DEFINE AND TEST MODELS

# 7.1 Define models
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('LR', LogisticRegression(
    solver='liblinear', multi_class='ovr')))
models.append(('xgb', xgb.XGBClassifier(subsample=0.2, min_samples_split=75,
                                        max_features='sqrt', n_estimators=8)))
models.append(('LDA', LinearDiscriminantAnalysis()))


# 7.2 evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.StratifiedKFold(
        n_splits=5, random_state=seed, shuffle=True)  # Cross-validation definition
    cv_results = model_selection.cross_val_score(
        model, train_X, train_y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# 8 CHOOSE AND USE MODEL
LDA = LinearDiscriminantAnalysis(solver='lsqr')
LDA.fit(train_X, train_y)
predictions = LDA.predict(val_X)
percent_predictions = LDA.predict_proba(val_X)[:, 1]
percent_predictions_r = percent_predictions.round(decimals=3)
print('Predicted class, 0=no seat, 1 = seat')
print(predictions[0:20])
print('Probability of getting elected')
print(percent_predictions_r[0:20])

print("LDA")
print(accuracy_score(val_y, predictions))
print(confusion_matrix(val_y, predictions))
print(classification_report(val_y, predictions))

# Precision = how many of positive labels were actually positive -> percent?, how useful model is?
# Recall = how many of positive instances did it classify as positive -> percent?, how complete results are?
#F1 = 2 * (precision*recall/precision+recall)
# The support is the number of occurrences of each class in y_true

# Xgb-model
xg_reg = xgb.XGBClassifier(subsample=0.2, min_samples_split=75,
                           max_features='sqrt', n_estimators=8)
xg_reg.fit(train_X, train_y)


preds = xg_reg.predict(val_X)
print("XgB-model")
print(confusion_matrix(val_y, preds))
print(classification_report(val_y, preds))

# See which features are the most important (F score is how many splits has been done on this metric)
xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()

# Lightgbm-model, actually only for datasets over 10 000 rows
dtrain = lgb.Dataset(train_X, label=train_y)
params = {}
params['objective'] = 'binary'
'''
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10
'''
clf = lgb.train(params=params, train_set=dtrain)
# is percentages so has to be converted to binary
lgb_pred = clf.predict(val_X)
lgb_pred_bin = lgb_pred.round()
print(classification_report(val_y, lgb_pred_bin))
