# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

df=pd.read_csv('/content/titanic_dataset.csv')

df.head()

df.isnull().sum()

df.drop('Cabin',axis=1,inplace=True)

df.drop('Name',axis=1,inplace=True)

df.drop('Ticket',axis=1,inplace=True)

df.drop('PassengerId',axis=1,inplace=True)

df.drop('Parch',axis=1,inplace=True)

df

df['Age']=df['Age'].fillna(df['Age'].median())

df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

df.isnull().sum()

plt.title("Dataset with outliers")

df.boxplot()

plt.show()

cols = ['Age','SibSp','Fare']

Q1 = df[cols].quantile(0.25)

Q3 = df[cols].quantile(0.75)

IQR = Q3 - Q1

df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

plt.title("Dataset after removing outliers")

df.boxplot()

plt.show()

from sklearn.preprocessing import OrdinalEncoder

climate = ['C','S','Q']

en= OrdinalEncoder(categories = [climate])

df['Embarked']=en.fit_transform(df[["Embarked"]])

df

climate = ['male','female']

en= OrdinalEncoder(categories = [climate])

df['Sex']=en.fit_transform(df[["Sex"]])

df

from sklearn.preprocessing import RobustScaler

sc=RobustScaler()

df=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])

df

import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)

df1=pd.DataFrame()

df1["Survived"]=np.sqrt(df["Survived"])

df1["Pclass"],parameters=stats.yeojohnson(df["Pclass"])

df1["Sex"]=np.sqrt(df["Sex"])

df1["Age"]=df["Age"]

df1["SibSp"],parameters=stats.yeojohnson(df["SibSp"])

df1["Fare"],parameters=stats.yeojohnson(df["Fare"])

df1["Embarked"]=df["Embarked"]

df1.skew()

import matplotlib

import seaborn as sns

import statsmodels.api as sm

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X = df1.drop("Survived",1)

y = df1["Survived"]

plt.figure(figsize=(12,10))

cor = df1.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)

plt.show()

cor_target = abs(cor["Survived"])

relevant_features = cor_target[cor_target>0.5]

relevant_features

X_1 = sm.add_constant(X)

model = sm.OLS(y,X_1).fit()

model.pvalues

cols = list(X.columns)

pmax = 1

while (len(cols)>0):
  p= []
  
  X_1 = X[cols]

  X_1 = sm.add_constant(X_1)

  model = sm.OLS(y,X_1).fit()

  p = pd.Series(model.pvalues.values[1:],index = cols)  

  pmax = max(p)

  feature_with_p_max = p.idxmax()

  if(pmax>0.05):

      cols.remove(feature_with_p_max)
    
  else:

    break
selected_features_BE = cols

print(selected_features_BE)

model = LinearRegression()

rfe = RFE(model,step= 4)

X_rfe = rfe.fit_transform(X,y)

model.fit(X_rfe,y)

print(rfe.support_)

print(rfe.ranking_)

nof_list=np.arange(1,6)

high_score=0

nof=0

score_list =[]

for n in range(len(nof_list)):
  X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)

  model = LinearRegression()

  rfe = RFE(model,step=nof_list[n])

  X_train_rfe = rfe.fit_transform(X_train,y_train)

  X_test_rfe = rfe.transform(X_test)

  model.fit(X_train_rfe,y_train)

  score = model.score(X_test_rfe,y_test)

  score_list.append(score)

if(score>high_score):

    high_score = score
    
    nof = nof_list[n]
print("Optimum number of features: %d" %nof)

print("Score with %d features: %f" % (nof, high_score))


cols = list(X.columns)

model = LinearRegression()

rfe = RFE(model, step=2)

X_rfe = rfe.fit_transform(X,y)

model.fit(X_rfe,y)

temp = pd.Series(rfe.support_,index = cols)

selected_features_rfe = temp[temp==True].index

print(selected_features_rfe)

reg = LassoCV()

reg.fit(X, y)

print("Best alpha using built-in LassoCV: %f" % reg.alpha_)

print("Best score using built-in LassoCV: %f" %reg.score(X,y))

coef = pd.Series(reg.coef_, index = X.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)) + " variables")

imp_coef = coef.sort_values()

import matplotlib

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Feature importance using Lasso Model")

plt.show()



# OUPUT


![ex7 9](https://github.com/Bala1511/Ex-07-Feature-Selection/assets/118680410/40d8d549-acd6-4fb5-b667-a2757d412244)
![ex7 8](https://github.com/Bala1511/Ex-07-Feature-Selection/assets/118680410/dfd28fbb-32dd-4598-9e5e-aa1b3f991b4c)
![ex7 7](https://github.com/Bala1511/Ex-07-Feature-Selection/assets/118680410/d2d3d172-65ab-4416-83a2-084d02105820)
![ex7 6](https://github.com/Bala1511/Ex-07-Feature-Selection/assets/118680410/1e87e445-acdd-4a4a-90ca-7daf308ab9c1)
![ex7 5](https://github.com/Bala1511/Ex-07-Feature-Selection/assets/118680410/183e8b29-0280-4b02-9c3a-a558faa86dc4)
![ex7 4](https://github.com/Bala1511/Ex-07-Feature-Selection/assets/118680410/b12b8713-7985-47cb-9327-fa7c614f5797)
![ex7 3](https://github.com/Bala1511/Ex-07-Feature-Selection/assets/118680410/7700108f-898a-4379-8f22-b0d18f9faa11)
![ex7 2](https://github.com/Bala1511/Ex-07-Feature-Selection/assets/118680410/d20de563-9761-4059-8759-c8b517cf5d59)
![ex7 1](https://github.com/Bala1511/Ex-07-Feature-Selection/assets/118680410/87286ea3-b4f1-4d10-b39a-4d773d46f558)
![ex7 15](https://github.com/Bala1511/Ex-07-Feature-Selection/assets/118680410/97884004-4cd3-4850-900d-a0c870ed9ce3)
![ex7 14](https://github.com/Bala1511/Ex-07-Feature-Selection/assets/118680410/0f70fb39-8f91-4a2f-bdce-2397cc02da38)
![ex7 13](https://github.com/Bala1511/Ex-07-Feature-Selection/assets/118680410/c9caa5c5-e787-483d-8d21-4244ac4eaf54)
![ex7 12](https://github.com/Bala1511/Ex-07-Feature-Selection/assets/118680410/d16a6d07-dd7d-40af-9bfb-9a8fff6b88cd)
![ex7 11](https://github.com/Bala1511/Ex-07-Feature-Selection/assets/118680410/c084fd25-d937-4d92-abe2-4995aa99fbe9)
![ex7 10](https://github.com/Bala1511/Ex-07-Feature-Selection/assets/118680410/6671b4aa-5bf4-478d-a896-d025915975af)
