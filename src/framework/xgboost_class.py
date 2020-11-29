import category_encoders as ce
from calibrator import *
from predictor import *
import numpy as np
import xgboost as xgb


class XGB(main_model):

	def __init__(self, data, col_num):
		super().__init__(data, col_num)
		self.model = xgb.XGBRegressor(n_estimators=100, objective='reg:pseudohubererror')


	def _feature_engineering(self, data):
		print('great')
		data_X, data_Y = super()._feature_engineering(data)

		# growth rates:
		data_X['g_6'] = (data_X['-1'] - data_X['-6']) / data_X['-6']
		data_X['g_12'] = (data_X['-1'] - data_X['-12']) / data_X['-12']
		data_X['g_24'] = (data_X['-1'] - data_X['-24']) / data_X['-24']

		# target encoding
#		encoder = ce.TargetEncoder(cols=['brand', 'country', 'therap'])
#		encoder.fit(data_X, data_Y)
#		data_X = encoder.transform(data_X)

		# seasonality for months:
		data_X['month_entry'] = data_X['month_entry'].map({
			'Jan': 1,
			'Feb': 2,
			'Mar': 3,
			'Apr': 4,
			'May': 5,
			'Jun': 6,
			'Jul': 7,
			'Aug': 8,
			'Sep': 9,
			'Oct': 10,
			'Nov': 11,
			'Dec': 12
		})
		#data_X['sin_month'] = np.sin(2 * np.pi * data_X.month_entry / 12)
		#data_X['cos_month'] = np.cos(2 * np.pi * data_X.month_entry / 12)
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
