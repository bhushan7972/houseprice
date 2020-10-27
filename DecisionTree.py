import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import sklearn

housing=pd.read_csv('data.csv')
#print(housing.info())
#print(housing.columns)
#print(housing.describe())

from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
print("Total Train data:",len(train_set),'\n Total Test data',len(test_set))

from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,random_state=42,test_size=0.2)

#on spacfic columns wise data can shuffule
for train_index,test_index in split.split(housing,housing['CHAS']):
    strat_train_index=housing.loc[train_index]
    strat_test_index=housing.loc[test_index]
print(strat_test_index['CHAS'].value_counts(),strat_train_index['CHAS'].value_counts())

housing=strat_train_index.copy()

corr_matrix=housing.corr()

print(corr_matrix['MEDV'].sort_values(ascending=False))

'''
attribute=['MEDV','RM','ZN','LSTAT']
#scatter_matrix(housing[attribute],figsize=(12,8))

housing.plot(kind='scatter',x='RM',y='MEDV',alpha=0.8)
plt.show()
'''


housing=strat_train_index.drop("MEDV",axis=1)
housing_labels=strat_train_index['MEDV'].copy()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
my_pip=Pipeline(    [
    ('imputer',SimpleImputer(strategy='median')),
    ('std_scaller',StandardScaler())
    ])

new2_houshing=my_pip.fit_transform(housing)
print(new2_houshing.shape)

from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor()

model.fit(new2_houshing,housing_labels)
some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]
prepared_data=my_pip.transform(some_data)
print('predicted data',model.predict(prepared_data))
print('actual data:',list(some_labels))

from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(new2_houshing)
mse=mean_squared_error(housing_labels,housing_predictions)
rmse=np.sqrt(mse)
print('Mean Error:',mse,'\nMean Square Error',rmse)

from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,new2_houshing,housing_labels,scoring='neg_mean_squared_error',cv=10)
rmse_score=np.sqrt(-scores)
print('Cross Valdiation Score:',rmse_score)

def print_scores(scores):
    print('Scores:',scores)
    print('Mean',scores.mean())
    print('Standard deviation',scores.std())

print_scores(rmse_score)

#testing


x_test=strat_test_index.drop('MEDV',axis=1)
y_test=strat_test_index['MEDV'].copy()
x_test_prepared=my_pip.transform(x_test)
final_predixtion=model.predict(x_test_prepared)
final_mse=mean_squared_error(y_test,final_predixtion)
final_rmse=np.sqrt(final_mse)
print('final rmse:',final_rmse)
#print(final_predixtion,list(y_test))


from joblib import dump,load
dump(model,'DecisionTree.joblib')