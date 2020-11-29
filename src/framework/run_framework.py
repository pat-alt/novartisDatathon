from calibrator_GLM import calibrator_glm
from calibrator import create_classifyer, main_model
from xgboost_class import XGB


if __name__ == '__main__':
	for i in range(24):
		print(i)
		#create_classifyer('C:\\Users\\EGimenez\\ME\\projects\\BGSE\\Novartis', 'dt_merged_w.csv', main_model, 'helloworld_22_10', i, testing=0.5)
		#create_classifyer('C:\\Users\\EGimenez\\ME\\projects\\BGSE\\Novartis', 'dt_merged_w.csv', calibrator_glm, 'calibrator_glm_3', i, testing=0)
		create_classifyer('../..', 'dt_merged_w.csv', XGB, 'xgboost', i, testing=0.15)

