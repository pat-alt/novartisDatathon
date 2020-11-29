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
from metrix import apply_metrics


def run_predictor_clean(root_path, model_name):
	sub_file = pd.read_csv(root_path + '/data/submission_template.csv')
	original_data = pd.read_csv(root_path + '/data/dt_merged_w.csv')

	data_P = []
	data_L = []
	data_U = []

	for i in range(24):
		print(i)
		with open(root_path + '/models/'+model_name+'/' + 'model_'+str(i)+'.pk', 'rb') as f:
			nc = pickle.load(f)

			if i > 0:
				nc.my_data_update(data_P, data_L, data_U) # <- [0, i-1] unnormalized + [country, brand]

			data_P, data_L, data_U = nc.my_predict() # -> [0, i] unnormailize + [country, brand]
		try:
			sub_file = nc.update_sub_file(sub_file, original_data)
		except:
			print('hola')

	sub_file.to_csv(root_path + '/data/submission_template_'+model_name+'.csv', index=False)

	print('We will win!!!! ;-)')
	result = apply_metrics(sub_file)
	print('-)')

if __name__ == '__main__':
	#run_predictor_clean('C:\\Users\\EGimenez\\ME\\projects\\BGSE\\Novartis', 'helloworld')
	#run_predictor_clean('C:\\Users\\EGimenez\\ME\\projects\\BGSE\\Novartis', 'calibrator_glm_3')
	run_predictor_clean('/Users/simonneumeyer/Desktop/NOVARTIS/novartisDatathon', 'xgboost')
