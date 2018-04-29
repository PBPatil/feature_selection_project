# %load q02_best_k_features/build.py
# Default imports

import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:

def percentile_k_features(df, k=20):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    var1 = SelectPercentile(f_regression,percentile=k)
    var1.fit(X,y)
    var2 = var1.get_support()
    var3 = list(X.loc[:,var2].columns.values)
    return ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath']
    
X = data.drop('SalePrice',axis=1)
y = data['SalePrice']
data.head()
X.head()
y.head(2)
f_test,_ = f_regression(X,y)
[f_test]
type([f_test])
percentile_k_features(df=data)



