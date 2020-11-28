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



def run_predictor(root_path, model_name):
	sub_file = pd.read_csv(root_path + '/data/submission_template.csv')

	with open(root_path + '/models/' + model_name + '/' + 'model_' + str(23) + '.pk', 'rb') as f:
		nc = pickle.load(f)
	data_X_all = nc.data_X  # CONTRACT nc.data_X // The pandas has all the feature engineering and scaling
	index_test = data_X_all['test']==1.0  # CONTRACT nc.data_X
	data_X_all = data_X_all[index_test]

	with open(root_path + '/models/' + model_name + '/' + 'model_' + str(0) + '.pk', 'rb') as f:
		nc = pickle.load(f)
	data_X = nc.data_X
	data_X = data_X[index_test]
	future_columns_empty = nc.ifuture_columns_empty
	future_columns_empty = future_columns_empty[index_test]

	for i in range(24):
		print(i)
		with open(root_path + '/models/'+model_name+'/' + 'model_'+str(i)+'.pk', 'rb') as f:
			nc = pickle.load(f)

		aux = pd.DataFrame()
		aux['realized'] = data_X_all[str(i)]
		aux['predicted'] = nc.predict(data_X) # CONTRACT: predict(data_X) data_X = nc.data_X
		aux['future_empty'] = future_columns_empty[str(i)]

		data_X[str(i)] = aux.apply(lambda x: x['realized'] if x['future_empty'] != -1 else x['predicted'], axis=1)
		#data_X[str(i)] = aux['predicted']

		# Transforming the file
		try:
			sub_file = nc.merge_data(sub_file, data_X, i)
		except:
			print('hola')

	sub_file.to_csv(root_path + '/data/submission_template_'+model_name+'.csv')

	print('hola')

def run_predictor_clean(root_path, model_name):
	sub_file = pd.read_csv(root_path + '/data/submission_template.csv')

	data_P = []

	for i in range(24):
		print(i)
		with open(root_path + '/models/'+model_name+'/' + 'model_'+str(i)+'.pk', 'rb') as f:
			nc = pickle.load(f)

			if i > 0:
				nc.my_data_update(data_P) # <- [0, i-1] unnormalized + [country, brand]

			data_P = nc.my_predict() # -> [0, i] unnormailize + [country, brand]
		try:
			sub_file = nc.update_sub_file(sub_file)
		except:
			print('hola')

	sub_file.to_csv(root_path + '/data/submission_template_'+model_name+'.csv')

	print('hola')


if __name__ == '__main__':
	run_predictor_clean('C:\\Users\\EGimenez\\ME\\projects\\BGSE\\Novartis', 'helloworld')
