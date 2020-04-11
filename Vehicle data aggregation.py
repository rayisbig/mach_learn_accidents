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

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


# ### Import vehicle data

# In[ ]:


vehicle_filepath = "C:\\Users\\rcaldwell\\Documents\\modelling-club-team-1\\data\\Vehicles\\"
vehicle_files = os.listdir("C:\\Users\\rcaldwell\\Documents\\modelling-club-team-1\\data\\Vehicles")


# In[ ]:


# Read in vehicle data
vehicle_data = pd.read_csv(vehicle_filepath+vehicle_files[0])

for x in vehicle_files[1:len(vehicle_files)]:
    vehicle_data = vehicle_data.append(
        pd.read_csv(vehicle_filepath+x)
    )


# In[ ]:


vehicle_data.head(10)


# In[ ]:


vehicle_data = vehicle_data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])


# ### Feature encoding

# In[ ]:


def func_name(f):
    return(f.__name__)




def encode_features(data_frame,*list_of_encodings):
    function_list = {}
    #function_list['c_d_prodsum'] = (x['c'] * x['d']).sum()
    
    for x in list_of_encodings:
        x(data_frame,function_list)
    
    
    return pd.Series(function_list, index=func_name(*list_of_encodings))


# In[ ]:


list_of_encodings = []  


# #### Sex of driver

# In[ ]:


def sex_of_drivers(data_frame,function_list):
    # Count of male drivers
    function_list['count_male_drivers'] = data_frame['Sex_of_Driver'].apply(lambda x: x[x['Sex_of_Driver'] == 1]['Sex_of_Driver'].count())
    
    # Count of female drivers
    #function_list['count_female_drivers'] = data_frame['Sex_of_Driver'].apply(lambda x: x[x['Sex_of_Driver'] == 0].count())
  

list_of_encodings.append(sex_of_drivers)


# In[ ]:





# In[ ]:





# #### Apply all aggregations

# In[ ]:


agg_vehicle = vehicle_data.head().groupby('Accident_Index').apply(encode_features, *list_of_encodings)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# Test function - works
vehicle_data.head(10).groupby('Accident_Index').apply(lambda x: x[x['Sex_of_Driver'] == 1]['Sex_of_Driver'].count())

