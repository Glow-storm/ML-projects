from sklearn.ensemble import RandomForestClassifier
# here we random forst classifer not regressor as classifrr gives a discrete value and not continois value like a
# a regressor , we use classifer when we need answer like yes or no but for probality and contious values
# we use regressor
from sklearn.pipeline import  Pipeline,make_pipeline
from sklearn.model_selection import cross_val_score

import  pandas as pd



credit_data=pd.read_csv(r"D:\Downloads\Compressed\aer credit card\AER_credit_card_data.csv",true_values=['yes'],false_values=['no'] )
# the true values mean that yes will be taken as true and no as false

y=credit_data.card

x=credit_data.drop('card',axis=1)

print(":the number of rows in data",x.shape[0])
print(x.head())
# here there is no need of a pipeline but it is used a best practice plus we can use make_pipline to create a pipeline
pipeline1 = make_pipeline(RandomForestClassifier(random_state=0,n_estimators=100))
cv_score = cross_val_score(pipeline1,x,y,scoring='accuracy',cv=5)

print("cross validation score :",cv_score.mean())
# mean gives us the average of something

# Drop leaky predictors from dataset
potential_leaks = ['expenditure', 'share', 'active', 'majorcards']
X2 = x.drop(potential_leaks, axis=1)

# Evaluate the model with leaky predictors removed
cv_scores = cross_val_score(pipeline1, X2, y,
                            cv=5,
                            scoring='accuracy')

print("Cross-val accuracy: %f" % cv_scores.mean())
