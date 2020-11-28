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
import numpy as np
import scipy.special as sc

class main_model():
	def __init__(self, data, col_num):
		self.model = DecisionTreeRegressor()
		self.col_num = col_num
		self.data_souce = self._prep_data(data)
		self.avg = None
		self.ifuture_columns_empty = None
		self.pipe = None
		self.pipe_categ = None
		self.pipe_numbers = None
		self.data_L = self._prep_data(data)
		self.data_U = self._prep_data(data)

	def my_fit(self):
		will_keep = (self.data_souce[str(self.col_num)] != -1)
		data_norm = self._normalize()
		data_norm = data_norm[will_keep]
		data_features_X, data_Y = self._feature_engineering(data_norm)
		self._make_pipe(data_features_X)
		data_X = self._run_pipe(data_features_X)
		self.model.fit(data_X, data_Y)

	def my_predict(self):
		data_norm = self._normalize()
		data_features_X, data_Y = self._feature_engineering(data_norm)
		data_X = self._run_pipe(data_features_X)
		Y = self.model.predict(data_X)
		Y[Y<0.0001] = 0.0001
		L, U = self._get_confidence_int(Y)
		Y, L, U = self._de_normalize(Y, L, U)
		self._merge_pred_real(Y, L, U)

		return self.data_souce, self.data_L, self.data_U


	def _merge_pred_real(self, Y, L, U):
		aux = pd.DataFrame()
		aux['realized'] = self.data_souce[str(self.col_num)]
		aux['predicted'] = Y
		aux['predicted_L'] = L
		aux['predicted_U'] = U
		aux['future_empty'] = self.data_souce[str(self.col_num)].fillna(-1)

		self.data_souce[str(self.col_num)] = \
			aux.apply(lambda x: x['predicted'] if x['future_empty'] == -1 else x['realized'], axis=1)
		self.data_L[str(self.col_num)] = \
			aux.apply(lambda x: x['predicted_L'] if x['future_empty'] == -1 else x['realized']*0.999, axis=1)
		self.data_U[str(self.col_num)] = \
			aux.apply(lambda x: x['predicted_U'] if x['future_empty'] == -1 else x['realized']*1.001, axis=1)

	def my_data_update(self, data_P, data_L, data_U):
		col_nums = [str(i) for i in range(self.col_num)]
		self.data_souce[col_nums] = data_P[col_nums]
		self.data_L = data_L
		self.data_U = data_U

	def update_sub_file(self, sub_file):
		data_X = self.data_souce
		data_L = self.data_L
		data_U = self.data_U
		col_num = self.col_num
		for c in sub_file['country'].unique():
			for b in sub_file[sub_file['country']==c]['brand'].unique():
				aux = data_X[data_X['brand']==b]
				aux_L = data_L[data_L['brand']==b]
				aux_U = data_U[data_U['brand'] == b]
				aux = aux[aux['country']==c]
				aux_L = aux_L[aux_L['country'] == c]
				aux_U = aux_U[aux_U['country'] == c]

				try:
					v = aux[str(col_num)].values[0]
					l = aux_L[str(col_num)].values[0]
					u = aux_U[str(col_num)].values[0]
				except:
					print('hola')

				f_index = sub_file[(sub_file['country']==c) & (sub_file['brand']==b) & (sub_file['month_num']==col_num)]
				sub_file.loc[f_index.index, 'prediction'] = v
				sub_file.loc[f_index.index, 'pred_95_low'] = l
				sub_file.loc[f_index.index, 'pred_95_high'] = u

		return sub_file

	def _prep_data(self, data):
		# This is not a general function anymore
		numbers = data.columns[(data.dtypes == int) | (data.dtypes == float)]
		data['A']= data['A'].fillna(0)
		data['B']= data['B'].fillna(0)
		data['C']= data['C'].fillna(0)
		data['D']= data['D'].fillna(0)
		data[numbers] = data[numbers].fillna(-1)

		data['num_generics'] = data['num_generics'].map(lambda x: float(x))
		data['test'] = data['test'].map(lambda x: float(x))

		# Ignoring Future data
		my_columns = []
		for c in data.columns:
			try:
				c = int(c)
				if c <= self.col_num:
					my_columns.append(str(c))
			except:
				my_columns.append(c)

		data = data[my_columns]

		return data

	def _normalize(self):
		avg_cols = [str(i) for i in range(-12, 0)]

		aux = self.data_souce[avg_cols]
		aux = aux.mean(axis=1)

		norm_data = self.data_souce.copy()

		try:
			for i in range(-137, 24):
				norm_data[str(i)] = norm_data[str(i)] / aux
		except:
			pass

		self.avg = aux

		return norm_data

	def _de_normalize(self, Y, L, U):
		return Y * self.avg, L * self.avg, U * self.avg

	def _feature_engineering(self, data):
		col_num = self.col_num
		# Here we can do more feature engineering

		data = data.drop(['test'], axis=1)

		data_X = data.drop([str(col_num)], axis=1)
		data_Y = data[str(col_num)]

		return data_X, data_Y

	def _make_pipe(self, data_X):

		categ = data_X.columns[(data_X.dtypes != int) & (data_X.dtypes != float)]
		numbers = data_X.columns[(data_X.dtypes == int) | (data_X.dtypes == float)]
		numbers = list(numbers)

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
		preprocessor.fit(data_X)

		self.pipe = preprocessor
		self.pipe_categ = categ
		self.pipe_numbers = numbers

	def _run_pipe(self, data):
		my_column_name = data.columns
		X_train_proposed = self.pipe.transform(data)
		numbers = self.pipe_numbers
		categ = self.pipe_categ

		# Turn back into DataFrames:
		try:
			X_train_proposed = pd.DataFrame.sparse.from_spmatrix(
				X_train_proposed,
				columns=list(numbers) + list(self.pipe.transformers_[1][1]['onehot'].get_feature_names(categ)))
		except AttributeError:
			X_train_proposed = pd.DataFrame(
				X_train_proposed,
				columns=list(numbers) + list(self.pipe.transformers_[1][1]['onehot'].get_feature_names(categ)))

		return X_train_proposed

	def _get_confidence_int(self, Y):
		beta = 5.0

		Y = np.array(Y)
		Z = (1 - Y)
		W = Y * beta

		alpha = W/Z

		L = sc.betaincinv(alpha, beta, .075)
		U = sc.betaincinv(alpha, beta, .925)

		index = Y > .9
		L[index] = .95*Y[index]
		U[index] = 1.05*Y[index]

		L[Y < 0.00009] = 0.00009

		index = U < Y
		U[index] = 1.05 * Y[index]

		index = Y < L
		L[index] = 0.95 * Y[index]


		return L, U





class nov_calibrator():
	def __init__(self, data, col_num):
		self.model = DecisionTreeRegressor()
		self.data_souce = data
		self.col_num = col_num
		self.data_X = None
		self.data_Y = None
		self.avg = None
		self.ifuture_columns_empty = self._future_columns_empty(data)

	def fit_yourself(self):
		data_X, data_Y = self.easy_data(self.col_num)
		self.model.fit(data_X, data_Y)

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

	def de_normalize(self, X_train):
		aux = self.avg
		try:
			for i in range(-137, 24):
				X_train[str(i)] = X_train[str(i)]*aux
		except:
			pass

		return X_train

	def _future_columns_empty(self, X_train):
		aux_cols = [str(i) for i in range(0, 24)]
		return X_train[aux_cols].fillna(-1)

	def easy_data(self, col_num):

		data = self.data_souce

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

class NewNov(nov_calibrator):

	def __init__(self, data, col_num):
		super().__init__(data, col_num)
		self.model = DecisionTreeRegressor()


	def fit_yourself(self):
		print('great')
		super().fit_yourself()

def create_classifyer(root_path, data_name, class_name, model_name, col_num):
	data = pd.read_csv(root_path + '/data/' + data_name)
	nc = class_name(data, col_num)
	nc.my_fit()

	# Make directory
	try:
		os.mkdir(root_path + '/models/'+model_name+'/')
	except:
		pass

	with open(root_path + '/models/'+model_name+'/'+ 'model_'+str(col_num)+'.pk', 'wb') as f:
		pickle.dump(nc, f)

