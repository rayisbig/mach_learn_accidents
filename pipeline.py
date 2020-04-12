# Import packages

import pandas as pd
import numpy as np
import os
from datetime import datetime
import category_encoders as ce
import xgboost as xgb
from sklearn.model_selection import train_test_split
from typing import Tuple
import seaborn as sns
import glob

# Separate data preparations
from data_prep_accidents import accident_data_prep

# Join 3 datasets
accident_filepath = "data/Accidents/"
data = accident_data_prep(accident_filepath)

# Encode string columns
cat_columns = []

for i, v in enumerate(data.dtypes):
    if v == np.dtype('O'):
        cat_columns.append(data.dtypes.index[i])

# Encode response manually
cat_columns.remove('Accident_Index')
encoder = ce.OrdinalEncoder(cols=cat_columns
                            ,handle_missing='na'
                            ,return_df=True
                            ,verbose=1)

encoder.fit(data)

# Transform variables
df = encoder.transform(data)


# Split data
train = df[df["Date"] <= pd.Timestamp("2018-06-30")]
holdout = df[~df.isin(train)]

_ignore_columns = ['Accident_Index',
                   'Accident_Severity',
                   'Date'
                  ]

train = train.drop(_ignore_columns, axis=1)
holdout = holdout.drop(_ignore_columns, axis=1)


# ## Build Model


# Get list of features

response_col = 'response'
model_features = train.columns.tolist()
model_features.remove(response_col)
model_features.remove('Time')




# Split datasets
trainSet = df.sample(frac=0.8)
testSet = df[~df.isin(trainSet)].dropna()

X_train = trainSet[model_features].values
y_train = trainSet[response_col].values

X_test = testSet[model_features].values
y_test = testSet[response_col].values



# Define Custom loss function for this project
def Qloss(predt : np.ndarray, dtrain : xgb.DMatrix) -> Tuple[str, float]:
    n = dtrain.num_row()

    response_matrix = np.ndarray(shape=(n,3), dtype=float)

    for i,v in enumerate(dtrain.get_label()):
        if v == 1:
            response_matrix[i,0] = v
        if v == 2:
            response_matrix[i,1] = v
        if v == 3:
            response_matrix[i,2] = v

    #print(predt.shape)
    one = 1 - response_matrix[predt == 0]
    two = 1 - response_matrix[predt == 1]
    three = 1 - response_matrix[predt == 2]

    alls = np.concatenate((one,two,three))

    return "QLossFunction",np.sqrt(np.sum(alls ** 2)/n)



# Prepare data for XGBoost Model
dtrain = xgb.DMatrix(data     =  X_train
                     ,label   =  y_train)

dtest = xgb.DMatrix (data = X_test
                     ,label = y_test)

# XGBoost parameters
xgbparams = {
    "booster": "gbtree",
    "tree_method": "hist",
    "grow_policy": "lossguide",
    "eta": 0.1,
    "seed": 2019,
    "subsample": 0.5,
    "colsample_by_tree": 0.5,
    "max_depth": 10,
    "min_child_weight": 10,
    "objective": "multi:softmax",
    "evals": "logloss",
    "evals_metric": "accuracy",
    "gamma": 0,
    "num_class": 3
}



# Train model
model1 = xgb.train(
         dtrain = dtrain
        ,params = xgbparams
        ,evals = [(dtrain,'train'),(dtest,'test')]
        ,num_boost_round = 2000
        ,early_stopping_rounds = 50
        ,verbose_eval = 1

)

model1.predict(dtest)



