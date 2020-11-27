from framework.calibrator import create_classifyer


if __name__ == '__main__':
	for i in range(24):
		print(i)
		create_classifyer('C:\\Users\\EGimenez\\ME\\projects\\BGSE\\Novartis', 'dt_merged_w.csv', 'helloworld', i)

