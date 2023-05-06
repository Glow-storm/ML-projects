from sklearn.ensemble import  RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

melb_data=pd.read_csv(r"E:\Downloads\Compressed\melbourne data 2 small\melb_data.csv")

y=melb_data.Price
x=melb_data.drop("Price", axis=1)
print(melb_data)

x_train, x_val , y_train , y_val = train_test_split(x,y,random_state=5, train_size=0.8, test_size=0.2)

col_with_missing=[col for col in x_train.columns if x_train[col].isnull().any()]
reduced_x_train=x_train.drop(col_with_missing,axis=1,inplace=True)
reduced_x_val = x_val.drop(col_with_missing,axis=1, inplace=True)

# this line runs a loop through the columns and takes the columns which have values which have less then 10 unique values
# and they must be of the type object

low_cardinal_col=[cname for cname in x_train.columns if x_train[cname].nunique() <10 and x_train[cname].dtype == 'object']

# this line select the columns with numbers in it
numerical_col = [cname for cname in x_train.columns if x_train[cname].dtype in ['int64', 'float64']]


# now we add the numerical and categorical columns togather
my_col=low_cardinal_col + numerical_col
x_train=x_train[my_col].copy()
x_val=x_val[my_col].copy()

print(x_train.head())
# this prints a list of the object type columns name
a = x_train.dtypes == object
object_cols = list(a[a].index)

print("categorical data")
print(object_cols)

def score_dataset(x_train2,x_val_2,y_train_2,y_val_2):
    model = RandomForestRegressor(n_estimators=100,random_state=5)
    model.fit(x_train2,y_train_2)
    predict=model.predict(x_val_2)
    mae = mean_absolute_error(y_val_2,predict)
    return mae

# this is the first approch where we drop the catergorical data
drop_x_train = x_train.select_dtypes(exclude=['object'])
drop_x_val = x_val.select_dtypes(exclude=['object'])
print("result of first approch")
print(score_dataset(drop_x_train,drop_x_val,y_train,y_val))


# this is the second approch to using catergorical data where we use ordinal encoding

o_x_train = x_train.copy()
o_x_val = x_val.copy()

ordinal_encoder = OrdinalEncoder()

o_x_train[object_cols] = ordinal_encoder.fit_transform(o_x_train[object_cols])
o_x_val[object_cols] = ordinal_encoder.transform(o_x_val[object_cols])

print("result of Second Apporch")
print(score_dataset(o_x_train,o_x_val,y_train,y_val))

# Now we use the third approch of using One Hot encoder , one hot encoding does not do well when there is a catergocial
# variable with more then 15 values

#We set handle_unknown='ignore' to avoid errors when the validation data contains classes
# that aren't represented in the training data, and setting sparse=False ensures that
# the encoded columns are returned as a numpy array (instead of a sparse matrix).

one_hot_encoder= OneHotEncoder(sparse=False, handle_unknown='ignore')
one_x_train= pd.DataFrame(one_hot_encoder.fit_transform(o_x_train[object_cols]))
one_x_val = pd.DataFrame(one_hot_encoder.transform(o_x_val[object_cols]))

# one hot removed indexes so we put them back
one_x_train.index = x_train.index
one_x_val.index = x_val.index

# now we remove the categorical columns from the data so only one hot encoding remains
num_x_train= x_train.drop(object_cols, axis= 1)
num_x_val = x_val.drop(object_cols, axis = 1)

# now we attach the one hot encoded columns with our data
one_x_train= pd.concat([one_x_train,num_x_train], axis =1)
one_x_val = pd.concat([one_x_val,num_x_val],axis=1)

print("Result of Third Aprroch")
print(score_dataset(o_x_train,o_x_val,y_train,y_val))

model= RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(one_x_train,y_train)
predict=model.predict(one_x_val)
mae=mean_absolute_error(y_val,predict)
print(mae)


# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: x_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
print(sorted(d.items(), key=lambda x: x[1]))



# to submit a file i need the x_val.index meaning the index of the verfication data and then the result as in SalePrice
# from the predict variable
# Save test predictions to file
output = pd.DataFrame({'Id': x_val.index,
                       'SalePrice': predict})
output.to_csv('submission.csv', index=False)
