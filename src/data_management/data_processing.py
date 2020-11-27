from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder,  scale
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
# Helper functions for different methods:
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import preprocessing
import os

def bayes_ridge(X_train, numbers, categ, max_iter=30, add_indicator=True):
    # Imputation for numerical cols:
    imputer=IterativeImputer(random_state=0, estimator=BayesianRidge(), max_iter=max_iter, add_indicator=add_indicator)
    X_train = make_pipe(X_train, imputer=imputer, numbers=numbers, categ=categ)
    return X_train

# Helper function to slit by type


