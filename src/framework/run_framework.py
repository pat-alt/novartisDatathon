from calibrator import create_classifyer
from calibrator import main_model
from calibrator_GLM import calibrator_glm
from xgboost_class import XGB


if __name__ == '__main__':
	for i in range(24):
		print(i)
		#create_classifyer('C:\\Users\\EGimenez\\ME\\projects\\BGSE\\Novartis', 'dt_merged_w.csv', main_model, 'helloworld', i)
		#create_classifyer('C:\\Users\\EGimenez\\ME\\projects\\BGSE\\Novartis', 'dt_merged_w.csv', calibrator_glm, 'calibrator_glm_3', i, testing=0)
		create_classifyer('..', 'dt_merged_w.csv', XGB, 'xgboost', i)

