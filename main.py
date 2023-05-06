# Importing necessary libraries
from sklearn.tree import DecisionTreeRegressor  # Regression tree
from sklearn.metrics import mean_absolute_error  # Error calculation tool
from sklearn.model_selection import train_test_split  # Splitting data into training and validation sets
from sklearn.ensemble import RandomForestRegressor  # Random forest for regression
import pandas as pd  # Data manipulation library

# Importing the Melbourne housing dataset
melbourne_data = pd.read_csv(r"E:\Downloads\Compressed\melbourne data\Melbourne_housing_FULL.csv")

# Removing null values from the data
melbourne_data = melbourne_data.dropna(axis=0)

# Printing the first 5 rows of the data
print(melbourne_data.head())

# Printing the summary of the data
print(melbourne_data.describe())

# Setting the target (Price) and the features
y = melbourne_data.Price
features = ['Distance', 'Rooms', 'Postcode', 'Propertycount', 'Bedroom2', 'Bathroom', 'Landsize', 'BuildingArea',
           'YearBuilt', 'Lattitude', 'Longtitude']
x = melbourne_data[features]

# Splitting the data into training and validation sets
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=5)

# Fitting a Decision Tree Regressor on the training data
melbourne_data_model = DecisionTreeRegressor(random_state=7)
melbourne_data_model.fit(train_x, train_y)

# Predicting the prices for validation data
val_predict = (melbourne_data_model.predict(val_x))

# Calculating the Mean Absolute Error for the validation set
print(mean_absolute_error(val_y, val_predict))

# Function to get the Mean Absolute Error for different number of leaf nodes
def get_mae(max_leaf_nodes, train_x, train_y, val_x, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=5)
    model = model.fit(train_x, train_y)
    val_predict2 = model.predict(val_x)
    mae = mean_absolute_error(val_y, val_predict2)
    return mae

# Testing the function for different number of leaf nodes
for max_leaf_nodes in [5, 10, 20, 50, 100, 500, 1000, 2000, 5000, 100000]:
    my_mae = get_mae(max_leaf_nodes, train_x, train_y, val_x, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" % (max_leaf_nodes, my_mae))

# Fitting a Random Forest Regressor on the training data
forest_model = RandomForestRegressor(random_state=6)

forest_model.fit(train_x,train_y)
forest_predict=forest_model.predict(val_x)
forest_mae=mean_absolute_error(val_y,forest_predict)
print(forest_mae)
