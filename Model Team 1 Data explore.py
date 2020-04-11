#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
from datetime import datetime
import category_encoders as ce
import xgboost as xgb
from sklearn.model_selection import train_test_split
from typing import Tuple

import seaborn as sns


# In[ ]:


filepath = "C:\\Users\\rcaldwell\\Documents\\modelling-club-team-1\\data\\Accidents\\"


# In[ ]:


files = os.listdir("C:\\Users\\rcaldwell\\Documents\\modelling-club-team-1\\data\\Accidents")


# In[ ]:


# Function to read in dates correctly
date_parser = lambda x : datetime.strptime(x, "%Y-%m-%d")
data = pd.concat([pd.read_csv(filepath+x
                   ,index_col=0
                   ,parse_dates=["date_new"]) for x in files]
                ,ignore_index=True)

data.head(10)


# ### Vehicles

# In[ ]:


vehicle_filepath = "C:\\Users\\rcaldwell\\Documents\\modelling-club-team-1\\data\\Vehicles\\"
vehicle_files = os.listdir("C:\\Users\\rcaldwell\\Documents\\modelling-club-team-1\\data\\Vehicles")


# In[ ]:


vehicle_files


# In[ ]:


# Function to read in dates correctly
vehicle_date_parser = lambda x : datetime.strptime(x, "%Y-%m-%d")
vehicle_data = pd.concat(pd.read_csv(vehicle_filepath + vehicle_files))

vehicle_data.head(10)


# In[ ]:


# Explore variables


# In[ ]:





# In[ ]:


# Aggregate correctly


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


data.info()


# Add total darkness flag

# In[ ]:


data['Light_Conditions'].unique()


# In[ ]:


data['dark_flag'] = pd.when(data['Light_Conditions'].isin('Darkness - lights unlit', 'Darkness - no lighting'),1)


# In[ ]:


data['Road_Surface_Conditions'].unique()


# In[ ]:


cat_columns = []

for i, v in enumerate(data.dtypes):
    if v == np.dtype('O'):
        cat_columns.append(data.dtypes.index[i])

# Encode response manually
cat_columns.remove('Accident_Severity')
cat_columns


# ### Variable Explore

# In[ ]:


np.sort(data['Police_Force'].unique())


# In[ ]:


data.groupby(['Number_of_Vehicles']).count()["Accident_Index"].plot.bar()


# In[ ]:


data.groupby(['Number_of_Casualties']).count()["Accident_Index"].plot.bar()


# In[ ]:


dowData = data.groupby(by=["Day_of_Week"]).count()["Accident_Index"]

dowData.plot.bar()


# In[ ]:


strToTime = lambda x: datetime.strptime(x, "%H:%M")

data['Time'] = [strToTime(str(x)) if type(x) != float else strToTime('00:00') for x in data["Time"]]


# In[ ]:


data['Time'].describe()


# In[ ]:


hours = pd.DataFrame( 
                        {"hours" : [x.hour if x!=np.nan else 0 for x in data['Time']],
                         "ind" : [1 for x in data['Time']]}
                    )\
                    .groupby('hours').count()

hours.plot.bar()


# In[ ]:


len(data['Local_Authority_(District)'].unique())


# In[ ]:


data    .groupby('Local_Authority_(District)')    .count()    .sort_values(by='Accident_Severity', ascending=False)    ['Accident_Severity']


# In[ ]:


len(data['Local_Authority_(Highway)'].unique())


# In[ ]:


data    .groupby('Local_Authority_(Highway)')    .count()    .sort_values(by='Accident_Severity', ascending=False)    ['Accident_Severity']


# In[ ]:


data['1st_Road_Class'].unique()


# In[ ]:


data['1st_Road_Number'].head()


# In[ ]:


data['Road_Type'].unique()


# In[ ]:


data['Speed_limit'].unique()


# In[ ]:


data['Junction_Detail'].unique()


# In[ ]:


data['Junction_Control'].unique()


# In[ ]:


data['2nd_Road_Class'].unique()


# ### Modelling file preparation

# In[ ]:


# encode variables
severity = {'Slight':0, 'Serious':1, 'Fatal':2}

data['response'] = [severity[x] for x in data['Accident_Severity']]
data['hour'] = [x.hour for x in data['Time']]

dowEncodings = pd.get_dummies(data, columns=['Day_of_Week'])
policeEncoding = pd.get_dummies(data, columns=['Police_Force'])

# Use binary encoding for Local_Authority_(District)

localAuthorityDistrictEncoder = ce.BinaryEncoder(cols=['Local_Authority_(District)'])
dfLocalAuthorityDistrict = localAuthorityDistrictEncoder.fit_transform(data['Local_Authority_(District)'])


# In[ ]:


encoder = ce.OrdinalEncoder(cols=cat_columns
                            ,handle_missing='na'
                            ,return_df=True
                            ,verbose=1)

encoder.fit(data)


# In[ ]:


# Transform variables
df = encoder.transform(data)

df.head()


# In[ ]:


# Split data
train = df[df["date_new"] <= pd.Timestamp("2010-12-31")]
holdout = df[~df.isin(train)]

_ignore_columns = ['Unnamed: 0.1',
                   'Accident_Index',
                   'Accident_Severity',
                   'date_new'
                  ]

train = train.drop(_ignore_columns, axis=1)
holdout = holdout.drop(_ignore_columns, axis=1)


# ## Build Model

# In[ ]:


# Get list of features
model_features = train.columns.tolist()
model_features.remove('response')

response = 'response'

# Split datasets
trainSet = df.sample(frac=0.8)
testSet = df[~df.isin(trainSet)].dropna()

X_train = trainSet[model_features].values
y_train = trainSet[response].values

X_test = testSet[model_features].values
y_test = testSet[response].values


# In[ ]:


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


# In[ ]:


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
    "evals_metric": "logloss",
    "gamma": 0,
    "num_class": 3
}


# In[ ]:


# Train model
model1 = xgb.train(
         dtrain = dtrain
        ,params = xgbparams
        ,evals = [(dtrain,'train'),(dtest,'test')]
        ,num_boost_round = 20
        ,early_stopping_rounds = 50
        ,verbose_eval = 1
        ,feval = Qloss
)


# In[ ]:


xgb.plot_importance(model1)

