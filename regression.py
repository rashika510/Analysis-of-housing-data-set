import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
from scipy import stats
from scipy.stats import norm, skew 
#limit values upto 3 decimal places
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) 

train = pd.read_csv('train.csv')
#############CLEANING###########
train_ID = train['Id']
#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)

fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
#deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])
#Now the data appears more normally distributed 

############FEATURE ENGINEERING############
y = train.SalePrice.values
X = train
X = X.drop(['SalePrice'], axis=1, inplace=True)
X_na = (X.isnull().sum() / len(X)) * 100
X_na = X_na.drop(X_na[X_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :X_na})
missing_data.head(20)

corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)

X["PoolQC"] = X["PoolQC"].fillna("None")
X["MiscFeature"] = X["MiscFeature"].fillna("None")
X["Alley"] = X["Alley"].fillna("None")
X["Fence"] = X["Fence"].fillna("None")
X["FireplaceQu"] = X["FireplaceQu"].fillna("None")

#lotfrontage by median of neighbourhood
X["LotFrontage"] = X.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
#garage
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    X[col] = X[col].fillna('None')
#cars in garage
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    X[col] = X[col].fillna(0)
#asement
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    X[col] = X[col].fillna(0)
#categorical basement
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    X[col] = X[col].fillna('None')
X["MasVnrType"] = X["MasVnrType"].fillna("None")
X["MasVnrArea"] = X["MasVnrArea"].fillna(0)
#general zoning by most recurring value
X['MSZoning'] = X['MSZoning'].fillna(X['MSZoning'].mode()[0])
#utilities wont ehlp so remove
X = X.drop(['Utilities'], axis=1)
X["Functional"] = X["Functional"].fillna("Typ")
X['Electrical'] = X['Electrical'].fillna(X['Electrical'].mode()[0])
X['KitchenQual'] = X['KitchenQual'].fillna(X['KitchenQual'].mode()[0])
X['Exterior1st'] = X['Exterior1st'].fillna(X['Exterior1st'].mode()[0])
X['Exterior2nd'] = X['Exterior2nd'].fillna(X['Exterior2nd'].mode()[0])
X['SaleType'] = X['SaleType'].fillna(X['SaleType'].mode()[0])
X['MSSubClass'] = X['MSSubClass'].fillna("None")

#MSSubClass=The building class
X['MSSubClass'] = X['MSSubClass'].apply(str)
#Changing OverallCond into a categorical variable
X['OverallCond'] = X['OverallCond'].astype(str)
#Year and month sold are transformed into categorical features.
X['YrSold'] = X['YrSold'].astype(str)
X['MoSold'] = X['MoSold'].astype(str)

from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(X[c].values)) 
    X[c] = lbl.transform(list(X[c].values))
#new useful column
X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
numeric_feats = X.dtypes[X.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = X[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    X[feat] = boxcox1p(X[feat], lam)
    
X = pd.get_dummies(X)
    
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)

##########Multiple Linear Regression###########
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
print('RMSE for Mulitple Linear Regression is {:.4f}'.format(sqrt(mean_squared_error(y_test, y_pred))))
regressor.score(X_test,y_test)
#############Decision Tree Regression##########
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train,y_train)

############RandomForestRegressor############
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300) 
regressor.fit(X_train,y_train)

############Gradient Boosting##############
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
GBoost.fit(X_train,y_train)
y_pred = GBoost.predict(X_test)
model_xgb.score(X_test,y_test)

############XGBoost##############
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_xgb.fit(X_train,y_train)
y_pred = model_xgb.predict(X_test)







