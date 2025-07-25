## Importing the modded dataset
import pandas as pd
dataset = pd.read_csv('dataset.csv')

## Dropping Redundant Variables
dataset = dataset.drop(columns=['DAILY_YIELD','TOTAL_YIELD','DATE','TIME','YEAR','MINUTES','TOTAL MINUTES PASS'])

## Spliting Data for train and test
X = dataset.drop(columns=['DC_POWER'])
y = dataset['DC_POWER']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, train_size=0.2,random_state=42)

## Pipelining
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',XGBRegressor(eta=0.04,max_depth=5))
])

pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)

## Metrics Evaluation
from sklearn.metrics import root_mean_squared_error,r2_score
print('RMSE:', root_mean_squared_error(y_test,y_pred)) # 211
print('R2:', r2_score(y_test,y_pred)) # 67.42%