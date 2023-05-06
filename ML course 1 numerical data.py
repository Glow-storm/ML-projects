# In this file we learn how to deal with missing values in our numerical  data
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import  SimpleImputer # this is used to impute new values where values are missing in data
# impute means to take the mean values of the columns and put it into the null values

melburne_data = pd.read_csv(r'E:\Downloads\Compressed\melbourne data 2 small\melb_data.csv')
y = melburne_data.Price
features = melburne_data.drop(['Price'], axis=1)
# here we take an alternative approch to selecting features by first
# taking all features except price and then in the second line we exclude all features with data type string or catergorical data
# **************************************************************************************************************
# Note  when using dropna when data remember not to store it in another variable or it wont work 4
# example x=melb_data.dropna(axis=0,inplace=True,subset=['price])  this is wrong and will not work
# use it like this instead  melb_data.dropna(axis=0,inplace=True,subset=['price]) , just dont store the result
# another variable and use it as it is
#***************************************************************************************************************
x = features.select_dtypes(exclude=['object'])

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, train_size=0.8, random_state=5)


# we can use train and test size in split to decide the ratio of train to test data
def score_dataset(x_train2, x_val2, y_train, y_val):  # this is a function that checks different datasets
    model = RandomForestRegressor(n_estimators=10, random_state=5)
    model.fit(x_train2, y_train)
    predict = model.predict(x_val2)
    return mean_absolute_error(y_val, predict)

# this was the first approch to dealing with missing values where we just drop all coulumns that have a missing value
col_with_missing = [col for col in x_train.columns if x_train[col].isnull().any()]  # this is a varible that takes columns
# with missing values
reduced_x_train = x_train.drop(col_with_missing, axis=1)
reduced_x_val = x_val.drop(col_with_missing, axis=1)
print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_x_train, reduced_x_val, y_train, y_val))


#This is the second approch where we impute a values in the columns where values are misssing

my_imputer=SimpleImputer()
imputed_x_train= pd.DataFrame(my_imputer.fit_transform(x_train))
imputed_x_val = pd.DataFrame(my_imputer.transform(x_val))
# imputation removed the columns name so we set them again
imputed_x_train.columns = x_train.columns
imputed_x_val.columns = x_val.columns

print("Result of imputation")
print(score_dataset(imputed_x_train,imputed_x_val,y_train,y_val))

# now we try the third approch with impuation extension where we mark the values that were missing
x_train_plus = x_train.copy() # we make a copy of the original so as to not change the original data
x_val_plus = x_val.copy()

# Make new columns indicating what will be imputed
for col in col_with_missing:
    x_train_plus[col + '_was_missing'] = x_train_plus[col].isnull()
    x_val_plus[col + '_was_missing'] = x_val_plus[col].isnull()

imputed_x_train2 = pd.DataFrame(my_imputer.fit_transform(x_train_plus))
imputed_x_val2 = pd.DataFrame(my_imputer.transform(x_val_plus))

print('Result of the third approch')
print(score_dataset(imputed_x_train2,imputed_x_val2,y_train,y_val))

# Shape of training data (num_rows, num_columns)
print(x_train.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (x_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
model= RandomForestRegressor(n_estimators=100,random_state=2)
model.fit(imputed_x_train2,y_train)
predict=model.predict(imputed_x_val2)


# Save test predictions to file
output = pd.DataFrame({'Id': x_val.index,
                       'SalePrice': predict})
output.to_csv('submission.csv', index=False)


