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

class nov_calibrator():
	def __init__(self, model):
		self.model = model
		self.data_X = None
		self.data_Y = None
		self.avg = None
		self.ifuture_columns_empty = None

	def fit(self, X, Y):
		self.model.fit(X, Y)

	def predict(self, X):
	    return self.model.predict(X)

	def make_pipe(self, X_train, numbers, categ):
		# This is not a general function anymore
		X_train['A']= X_train['A'].fillna(0)
		X_train['B']= X_train['B'].fillna(0)
		X_train['C']= X_train['C'].fillna(0)
		X_train['D']= X_train['D'].fillna(0)
		X_train[numbers] = X_train[numbers].fillna(-1)

		# We exclude test from numbers
		numbers = list(numbers)
		numbers.remove('test')

		test = X_train['test']

		# Split preprocessing depending on type:
		# 1) Numerical:
		numeric_transformer = Pipeline(
			steps=[
				('scaler', StandardScaler())
			]
		)
		# 2) Categorical:
		categorical_transformer = Pipeline(
			steps=[
				('onehot', OneHotEncoder(handle_unknown='ignore'))
			]
		)
		# Combine:
		preprocessor = ColumnTransformer(
			transformers=[
				('num', numeric_transformer, numbers),
				('cat', categorical_transformer, categ)
			],
			remainder='drop'
		)
		# Fit and transform:
		preprocessor.fit(X_train)
		my_column_name = X_train.columns
		X_train_proposed = preprocessor.transform(X_train)
		# Turn back into DataFrames:
		try:
			X_train_proposed = pd.DataFrame.sparse.from_spmatrix(
				X_train_proposed,
				columns=list(numbers) + list(preprocessor.transformers_[1][1]['onehot'].get_feature_names(categ)))
		except AttributeError:
			X_train_proposed = pd.DataFrame(
				X_train_proposed,
				columns=list(numbers) + list(preprocessor.transformers_[1][1]['onehot'].get_feature_names(categ)))

		X_train_proposed['test'] = test

		return X_train_proposed

	def normalize(self, X_train):

		avg_cols = [str(i) for i in range(-12, 0)]

		aux = X_train[avg_cols]
		aux = aux.mean(axis=1)

		try:
			for i in range(-137, 24):
				X_train[str(i)] = X_train[str(i)]/aux
		except:
			pass

		self.avg = aux

		return X_train

	def future_columns_empty(self, X_train):
		aux_cols = [str(i) for i in range(0, 24)]
		self.ifuture_columns_empty = X_train[aux_cols].fillna(-1)

	def easy_data(self, data, col_num):

		self.future_columns_empty(data)

		my_columns = []
		for c in data.columns:
			try:
				c = int(c)
				if c <= col_num:
					my_columns.append(str(c))
			except:
				my_columns.append(c)

		data = data[my_columns]
		will_drop = (data[str(col_num)].isna() == False)
		data = self.normalize(data)

		data_X = data.drop([str(col_num)], axis=1)
		data_Y = data[str(col_num)]

		data_X['num_generics'] = data_X['num_generics'].map(lambda x: float(x))
		data_X['test'] = data_X['test'].map(lambda x: float(x))

		categ = data_X.columns[(data_X.dtypes != int) & (data_X.dtypes != float)]
		numbers = data_X.columns[(data_X.dtypes == int) | (data_X.dtypes == float)]

		data_X = self.make_pipe(data_X, numbers, categ)
		# data_X = bayes_ridge(data_X, [], categ)

		self.data_X = data_X
		self.data_Y = data_Y

		data_X = data_X[will_drop]
		data_Y = data_Y[will_drop]

		return data_X, data_Y

	def merge_data(self, sub_file, data_X, col_num):
		for c in sub_file['country'].unique():
			for b in sub_file[sub_file['country']==c]['brand'].unique():
				aux = data_X[data_X['brand_'+b]==1.0]
				aux = aux[aux['country_'+c]==1.0]
				try:
					v = aux[str(col_num)].values[0]
				except:
					print('hola')

				f_index = sub_file[(sub_file['country']==c) & (sub_file['brand']==b) & (sub_file['month_num']==col_num)]
				sub_file.loc[f_index.index, 'prediction'] = v

		return sub_file

def create_classifyer(root_path, data_name, model_name, col_num):
	data = pd.read_csv(root_path + '/data/' + data_name)
	nc = nov_calibrator(DecisionTreeRegressor())
	data_X, data_Y = nc.easy_data(data, col_num)
	nc.fit(data_X, data_Y)

	# Make directory
	try:
		os.mkdir(root_path + '/models/'+model_name+'/')
	except:
		pass

	with open(root_path + '/models/'+model_name+'/'+ 'model_'+str(col_num)+'.pk', 'wb') as f:
		pickle.dump(nc, f)

