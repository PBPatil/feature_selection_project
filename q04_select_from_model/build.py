# %load q04_select_from_model/build.py
# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')


# Your solution code here
def select_from_model(df):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    rf = RandomForestClassifier(random_state=9)
    rf_sfm = SelectFromModel(rf)
    rf_sfm.fit(X,y)
    rf_sfm1 = rf_sfm.get_support()
    return list(X.loc[:,rf_sfm1].columns.values)


