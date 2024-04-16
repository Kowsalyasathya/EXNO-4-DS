# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
## FEATURE SCALLING
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
df.dropna()
max_vals = np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/Kowsalyasathya/EXNO-4-DS/assets/118671457/ad51e132-2952-403f-9cae-e0a97f237c41)
## STANDARD SCALING:
```
df1=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![image](https://github.com/Kowsalyasathya/EXNO-4-DS/assets/118671457/32c0438a-da8d-4227-8e98-d6d45a00fe5d)
## MIN-MAX SCALING:
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/Kowsalyasathya/EXNO-4-DS/assets/118671457/d17222da-e7e8-4b90-a2ef-7c03ba8dd5ad)
## NORMALIZATION:
```
df2=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2
```
![image](https://github.com/Kowsalyasathya/EXNO-4-DS/assets/118671457/ea12bbf6-308b-4690-9b7e-7148bab4bc9b)
## MAXIMUN ABSOLUTE SCALING:
```
df3=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![image](https://github.com/Kowsalyasathya/EXNO-4-DS/assets/118671457/0f473b8e-7d95-4fba-bba7-db0b178d6c45)

## ROBUST SCALER:
```
df4=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```
![image](https://github.com/Kowsalyasathya/EXNO-4-DS/assets/118671457/008b9a58-3c29-4ae0-bcb4-3134b1d7c032)
![image](https://github.com/Kowsalyasathya/EXNO-4-DS/assets/118671457/6163f656-108e-44f4-bfb5-c6b388030975)
## FEATURE SELECTION:
```
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv("/content/income(1) (1).csv")
df
```
### FILTER METHOD:
```
le = LabelEncoder()
for col in df.select_dtypes(include=['object']):
    df[col] = le.fit_transform(df[col])
X = df.drop(columns=['SalStat'])
y = df['SalStat']
selector = SelectKBest(score_func=chi2, k=5)
X_new = selector.fit_transform(X, y)
selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
for feature in selected_features:
    print("-", feature)
```
![image](https://github.com/Kowsalyasathya/EXNO-4-DS/assets/118671457/c859bbe5-0295-41a8-ae80-9c69c3a91ed3)
### WRAPPER METHOD:
```
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("/content/income(1) (1).csv")
le = LabelEncoder()
for col in df.select_dtypes(include=['object']):
    df[col] = le.fit_transform(df[col])
X = df.drop(columns=['SalStat'])
y = df['SalStat']
estimator = LogisticRegression()
rfe = RFE(estimator, n_features_to_select=5, step=1)
rfe.fit(X, y)
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
for feature in selected_features:
    print("-", feature)
```
![image](https://github.com/Kowsalyasathya/EXNO-4-DS/assets/118671457/231fcd9c-ee38-4cfa-826d-0629b44007fa)
### EMBEDDED METHOD:
```
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("/content/income(1) (1).csv")
df['SalStat'] = df['SalStat'].apply(lambda x: 0 if x == 'less than or equal to 50,000' else 1)
le = LabelEncoder()
for col in df.select_dtypes(include=['object']):
    df[col] = le.fit_transform(df[col])
X = df.drop(columns=['SalStat'])
y = df['SalStat']
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)
feature_importances = clf.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
selected_features = feature_importance_df.head(5)['Feature'].tolist()
print("Selected Features:")
for feature in selected_features:
    print("-", feature)
```
![image](https://github.com/Kowsalyasathya/EXNO-4-DS/assets/118671457/7a060b73-0853-425d-8f06-4572c89c4adf)

# RESULT:
Thus, Feature selection and Feature scaling has been performed using the given dataset.
