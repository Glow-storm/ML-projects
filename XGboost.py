from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import  pandas as pd
from xgboost import XGBRegressor


melburne_data = pd.read_csv(r'D:\Downloads\Compressed\melbourne data 2 small\melb_data.csv')

cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
x = melburne_data[cols_to_use]
y = melburne_data.Price

x_train, x_valid, y_train, y_valid = train_test_split(x,y)
# XGBregerrsor is a different type of modeling technique where we create a single model and add refined models on top of it
#
model = XGBRegressor(n_estimators=100,learning_rate=0.05,n_jobs=8)
# early_stopping_rounds offers a way to automatically find the ideal value for n_estimators.
# Early stopping causes the model to stop iterating when the validation score stops improving,
# even if we aren't at the hard stop for n_estimators.
# It's smart to set a high value for n_estimators and then use early_stopping_rounds to find the optimal time to stop iterating.
# When using early_stopping_rounds,
# you also need to set aside some data for calculating the validation scores - this is done by setting the eval_set parameter.

#learning_rate
#Instead of getting predictions by simply adding up the predictions from each component model,
# we can multiply the predictions from each model by a small number (known as the learning rate) before adding them in.
#This means each tree we add to the ensemble helps us less. So, we can set a higher value for n_estimators without overfitting.
# If we use early stopping, the appropriate number of trees will be determined automatically.
#In general, a small learning rate and large number of estimators will yield more accurate XGBoost models,
# though it will also take the model longer to train since it does more iterations through the cycle. As default,
# XGBoost sets learning_rate=0.1.

# n_jobs just decrease the time it uses to fit the model it is best to set it to the number of cores you have , but it will not
# help in creating a better model.
model.fit(x_train,y_train, early_stopping_rounds=5,eval_set=[(x_valid,y_valid)],verbose=False)

#learning_rate
#Instead of getting predictions by simply adding up the predictions from each component model,
# we can multiply the predictions from each model by a small number (known as the learning rate) before adding them in.
#This means each tree we add to the ensemble helps us less. So, we can set a higher value for n_estimators without overfitting.
# If we use early stopping, the appropriate number of trees will be determined automatically.
#In general, a small learning rate and large number of estimators will yield more accurate XGBoost models,
# though it will also take the model longer to train since it does more iterations through the cycle. As default,
# XGBoost sets learning_rate=0.1.

predict = model.predict(x_valid)
print("Mean Absolute Error  :"+ str(mean_absolute_error(y_valid,predict)))
