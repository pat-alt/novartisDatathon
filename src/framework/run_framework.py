from framework.calibrator import create_classifyer
from framework.calibrator import main_model
from framework.calibrator_GLM import calibrator_glm


if __name__ == '__main__':
	for i in range(24):
		print(i)
		create_classifyer('C:\\Users\\EGimenez\\ME\\projects\\BGSE\\Novartis', 'dt_merged_w.csv', main_model, 'helloworld_22_10', i, testing=0.5)
		#create_classifyer('C:\\Users\\EGimenez\\ME\\projects\\BGSE\\Novartis', 'dt_merged_w.csv', calibrator_glm, 'calibrator_glm_3_', i, testing=0)

