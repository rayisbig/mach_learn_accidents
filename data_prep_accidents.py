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


def accident_data_prep(filepath):
  # Import accident files

  accident_data = pd.concat([pd.read_csv(f) for f in  glob.glob(filepath + '*.csv')], ignore_index = True)

  # Fix dodgy types


  ### Clean column data

  # Speed Limit
  accident_data['Speed_limit'] = accident_data['Speed_limit'].fillna(value=-1).astype(np.int64)

  # Date
  strToDate = lambda x: datetime.strptime(x, "%d/%m/%Y")
  accident_data['Date'] = [strToDate(str(x)) for x in accident_data["Date"]]

  # Time
  strToTime = lambda x: datetime.strptime(x, "%H:%M")
  accident_data['Time'] = [strToTime(str(x)) if type(x) != float else strToTime('00:00') for x in accident_data["Time"]]

  # Hour
  accident_data['hour'] = [x.hour for x in accident_data['Time']]

  # Dow
  accident_data = pd.get_dummies(accident_data, columns=['Day_of_Week'])

  # Response
  #severity = {'Slight':0, 'Serious':1, 'Fatal':2}
  #accident_data['response'] = [severity[x] for x in accident_data['Accident_Severity']]
  accident_data['response'] = accident_data['Accident_Severity'] - 1

  # Police
  accident_data = pd.get_dummies(accident_data, columns=['Police_Force'])

  # Local authority
  localAuthorityDistrictEncoder = ce.BinaryEncoder(cols=['Local_Authority_(District)'])
  dfLocalAuthorityDistrict = localAuthorityDistrictEncoder.fit_transform(accident_data['Local_Authority_(District)'])
  accident_data = pd.concat([accident_data.reset_index(drop=True), dfLocalAuthorityDistrict], axis=1)


  return accident_data

