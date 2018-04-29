# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here

def rf_rfe(df):
    rf = RandomForestClassifier()
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    features = X.shape[1]/2
    rfe_rf = RFE(rf, n_features_to_select=features)
    rfe_rf.fit(X,y)
    return list(X.loc[:,rfe_rf.support_].columns.values)
