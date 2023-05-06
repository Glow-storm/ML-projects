from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import pandas as pd

melburne_data = pd.read_csv(r'D:\Downloads\Compressed\melbourne data 2 small\melb_data.csv')
y=melburne_data.Price
col_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
x=melburne_data[col_to_use]

my_pipline = Pipeline(steps=[("Preprocessor",SimpleImputer()),("model",RandomForestRegressor(n_estimators=100,random_state=0))])

# here we have to multiply witb -1 beacuse we get negative MAE so we turn it into positive
scores = -1 * cross_val_score(my_pipline,x,y,cv=5,scoring='neg_mean_absolute_error')
print("MAE scores:\n", scores)

