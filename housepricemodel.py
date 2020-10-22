import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import sklearn


housing=pd.read_csv('data.csv')
print(housing.info())
print(housing['CHAS'].value_counts())
print(housing.describe())

#housing.hist(bins=50,figsize=(20,15))
#plt.show()

###Train test split
'''
def split_train_test(data,test_ratio):
    np.random.seed(42)
    shuffle=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffle[:test_set_size]
    train_indices=shuffle[test_set_size:]

    return  data.iloc[train_indices],data.iloc[test_indices]

train_set,test_set=split_train_test(housing,0.2)

'''

from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
print('Total Traing data:',len(train_set),'\n Total test data:',len(test_set))

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
print(strat_test_set['CHAS'].value_counts())

housing=strat_train_set.copy()
#getting columns for opeartion


#corelation

corr_matrix=housing.corr()
print(corr_matrix['MEDV'].sort_values(ascending=False))

from pandas.plotting import scatter_matrix
attribute=['MEDV','RM','ZN','LSTAT']
#scatter_matrix(housing[attribute],figsize=(12,8))

housing.plot(kind='scatter',x='RM',y='MEDV',alpha=0.8)
plt.show()

#attribute combination

housing['TAXRM']=housing['TAX']/housing['RM']
print(housing.head())


corr_matrix=housing.corr()
print(corr_matrix['MEDV'].sort_values(ascending=False))

housing.plot(kind='scatter',x='TAXRM',y='MEDV',alpha=0.8)
plt.show()

housing=strat_train_set.drop("MEDV",axis=1)
housing_labels=strat_train_set['MEDV'].copy()
#Missing value
#1 remove only those nan row

a=housing.dropna(subset=['RM'])
print(a.shape)

#2 remove full attribute
housing.drop('RM',axis=1)
#3 rplace with 0,mean,median
median=housing['RM'].median()
housing['RM'].fillna(median)

#this code is fill all columns nan values

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='median')
imputer.fit(housing)
print(imputer.statistics_)

x=imputer.transform(housing)
new_houes=pd.DataFrame(x,columns=housing.columns)
print(new_houes.describe())

#feature scalling

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

my_pip=Pipeline(    [
    ('imputer',SimpleImputer(strategy='median')),
    ('std_scaller',StandardScaler())
    ])

#new2_houshing=my_pip.fit_transform(new_houes)
#print(new2_houshing)

new2_houshing=my_pip.fit_transform(housing)
print(new2_houshing.shape)


#SELECTING MODEL

#1 model 
from sklearn.linear_model import LinearRegression
model=LinearRegression()

#2model

from sklearn.tree import DecisionTreeRegressor
#model=DecisionTreeRegressor()

#3 model
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()

model.fit(new2_houshing , housing_labels)

some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]

prepared_data=my_pip.transform(some_data)
print('predicted data',model.predict(prepared_data))
print('actual data:',list(some_labels))

#EValuting model
from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(new2_houshing)
mse=mean_squared_error(housing_labels,housing_predictions)
rmse=np.sqrt(mse)
print(mse,rmse) # by use  liner r23.380136328422374 4.835301058716238
# by use desiontree 0.0 0.0

#Cross validation

from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,new2_houshing,housing_labels,scoring='neg_mean_squared_error',cv=10)
rmse_score=np.sqrt(-scores)
print(rmse_score)

def print_scores(scores):
    print('Scores:',scores)
    print('Mean',scores.mean())
    print('Standard deviation',scores.std())

print_scores(rmse_score)
'''
output:
decision tree:
Scores: [4.17772841 5.34696175 5.25865141 4.00524047 4.09914625 2.95076261
 4.97915656 3.79855236 3.45503256 4.47537708]
Mean 4.25466094552521
Standard deviation 0.7358418902203262

liner reg:
Scores: [4.22235612 4.26438649 5.09424333 3.83081183 5.37600331 4.41092152
 7.47272243 5.48554135 4.14606627 6.0717752 ]
Mean 5.037482786117751
Standard deviation 1.0594382405606948

RandomForestRegressor:
Scores: [2.80633555 2.8523119  4.41110118 2.73826674 3.49772225 2.69591409
 4.67984908 3.32570485 3.0675289  3.23117129]
Mean 3.33059058377952
Standard deviation 0.6597196212066484

'''

from joblib import dump,load
dump(model,'housepredictmodellib.joblib')

#Testing model

x_test=strat_test_set.drop('MEDV',axis=1)
y_test=strat_test_set['MEDV'].copy()
x_test_prepared=my_pip.transform(x_test)
final_predixtion=model.predict(x_test_prepared)
final_mse=mean_squared_error(y_test,final_predixtion)
final_rmse=np.sqrt(final_mse)
print('final rmse:',final_rmse)
print(final_predixtion,list(y_test))

print(prepared_data[0])