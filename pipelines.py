from sklearn.preprocessing import  OrdinalEncoder , OneHotEncoder
from sklearn.metrics import  mean_absolute_error
from  sklearn.ensemble import  RandomForestRegressor
from  sklearn.impute import  SimpleImputer
from  sklearn.compose import  ColumnTransformer
from  sklearn.pipeline import Pipeline
from  sklearn.model_selection import  train_test_split
import  pandas as pd

melb_data = pd.read_csv(r"D:\Downloads\Compressed\melbourne data 2 small\melb_data.csv")
y = melb_data.Price
x = melb_data.drop('Price',axis=1)

x_train_full, x_valid_full, y_train , y_valid=train_test_split(x,y,random_state=0,train_size=0.8,test_size=0.2)

num_cols= [col for col in x_train_full.columns if x_train_full[col].dtype in ['int64','float64']]



categorical_cols = [col for col in x_train_full.columns if x_train_full[col].nunique()<10 and x_train_full[col].dtype=='object']
my_features = num_cols+categorical_cols

x_train = x_train_full[my_features].copy()
x_valid = x_valid_full[my_features].copy()


# Preprocessing for numerical data
# here we make a imputer for the numerical data
num_transformer = SimpleImputer(strategy='constant')

# preprocessing for catergorical Data
# here we first make a imputater for the catergoircal data to fill any missing values and then we code the encoding method
# in this case we chose the OneHot encoding method
categorical_transformer = Pipeline(steps=[ ('imputer', SimpleImputer(strategy='most_frequent')),
                                            ('onehot',OneHotEncoder(handle_unknown="ignore")) ])

# Bundle preprocessing for numerical and categrical data togather
# here we just make a pipeline to join the two processes togather and run them
preprocessor = ColumnTransformer(transformers=[('num',num_transformer,num_cols),('cat',categorical_transformer,categorical_cols)])

model = RandomForestRegressor(n_estimators=100,random_state=0)
OrdinalEncoder(handle_unknown='ignore')
my_pipleine=Pipeline(steps=[("preprocessor",preprocessor),('model',model)])
my_pipleine.fit(x_train,y_train)
predict=my_pipleine.predict(x_valid)
print(mean_absolute_error(y_valid,predict))

