from sklearn.tree import DecisionTreeRegressor  #this the tree used to make predicitions
from sklearn.metrics import mean_absolute_error # this tool is used to detect errors in the result
from sklearn.model_selection import train_test_split # using this tool we split the data into training and validation data
from sklearn.ensemble import RandomForestRegressor # this tool is has greater effieceny then decision tree
import pandas as pd
# this is how we import a csv file in pycharm remember to put the 'r' before the address
melbourne_data=pd.read_csv(r"E:\Downloads\Compressed\melbourne data\Melbourne_housing_FULL.csv")
melbourne_data=melbourne_data.dropna(axis=0) # we use dropna to remove amy null values in the data
print(melbourne_data.head()) # using head allow us to get the top 5 values in the dataset
print(melbourne_data.describe()) # describe gives the infomartion about the dataset

y = melbourne_data.Price # this y is the target or the thing we want to predict in this case the price is the target
features=['Distance','Rooms','Postcode','Propertycount','Bedroom2','Bathroom','Landsize','BuildingArea','YearBuilt',
          'Lattitude','Longtitude'] # these are the features from which the ai will find patterns
# remember to only include the revelant features as unreavleant features can make wrong patterns
x = melbourne_data[features] # this x is the features or data which contains the features
#print(x.head())
#melbourne_data_model=DecisionTreeRegressor(random_state=5)
#melbourne_data_model.fit(x,y)
#print(melbourne_data_model.predict(x))

#guess_price=melbourne_data_model.predict(x)

#print(mean_absolute_error(y,guess_price))
# here i make the model and just use its own data to check it which is not right so now we create two datasets
# one for training and one for validition

train_x , val_x , train_y, val_y=train_test_split(x,y,random_state=5) # THE random state makes sure the data is split randomly

melbourne_data_model=DecisionTreeRegressor(random_state=7) # you can use any number for the random state
melbourne_data_model.fit(train_x,train_y)
val_predict=(melbourne_data_model.predict(val_x))
print(mean_absolute_error(val_y,val_predict))

# here we create a function which changes the number of leaf of a tree
# it you a too shallow tree then it wont recognizre patterns and have too many leafs and there will very less data in each leaf
# so we create a function that takes different no of leafs and gives the absolute error from which see which leaf setting is the best
def get_mae(max_leaf_nodes,train_x,train_y,val_x,val_y):
    model=DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=5)
    model=model.fit(train_x,train_y)
    val_predict2=model.predict(val_x)
    mae=mean_absolute_error(val_y,val_predict2)
    return mae
# here we use a loop to use the function we created
for max_leaf_nodes in [5,10,20,50,100,500,1000,2000,5000,100000]:
    my_mae=get_mae(max_leaf_nodes,train_x, train_y, val_x, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


forest_model=RandomForestRegressor(random_state=6) # here we use random forest instead of a single tree which reduces error
forest_model.fit(train_x,train_y)
forest_predict=forest_model.predict(val_x)
forest_mae=mean_absolute_error(val_y,forest_predict)
print(forest_mae)
