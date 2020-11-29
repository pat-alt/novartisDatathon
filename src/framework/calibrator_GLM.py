from calibrator import main_model
from sklearn.linear_model import TweedieRegressor

class calibrator_glm(main_model):
	def __init__(self, data, col_num):
		super().__init__(data, col_num)
		self.model = TweedieRegressor(power=2, link='log')
		self._spread =  0.01

	def _normalize(self):
		avg_cols = [str(i) for i in range(-12, 0)]

		aux = self.data_souce[avg_cols]
		aux = aux.mean(axis=1)

		norm_data = self.data_souce.copy()

		try:
			for i in range(-137, 24):
				norm_data[str(i)] = norm_data[str(i)] / aux + self._spread
		except:
			pass

		self.avg = aux

		return norm_data

	def _de_normalize(self, Y, L, U):
		return (Y - self._spread) * self.avg, (L - self._spread) * self.avg, (U - self._spread) * self.avg
