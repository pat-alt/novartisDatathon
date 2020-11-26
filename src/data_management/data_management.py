import pandas as pd

def get_data():
	gx_volume = pd.read_csv('../../data/gx_volume.csv')
	gx_therapeutic_area = pd.read_csv('../../data/gx_therapeutic_area.csv')
	gx_panel = pd.read_csv('../../data/gx_panel.csv')
	gx_package = pd.read_csv('../../data/gx_package.csv')
	gx_num_generics = pd.read_csv('../../data/gx_num_generics.csv')

	gx_volume = gx_volume.drop('Unnamed: 0', axis=1)
	gx_volume = gx_volume.merge(gx_therapeutic_area[['brand', 'therapeutic_area']], on='brand')
	gx_volume = gx_volume.merge(gx_panel[['country', 'brand', 'channel', 'channel_rate']], on=['country', 'brand'])
	gx_volume = gx_volume.merge(gx_package[['country', 'brand', 'presentation']], on=['country', 'brand'])
	gx_volume = gx_volume.merge(gx_num_generics[['country', 'brand', 'num_generics']], on=['country', 'brand'])

	gx_volume_norm = gx_volume[gx_volume['month_num'] == -1][['country', 'brand', 'volume']].rename(columns={"volume": "volume_1"})

	gx_volume = gx_volume.merge(gx_volume_norm, on=['country', 'brand'])
	gx_volume['ratio'] = gx_volume.apply(lambda x: x.volume / x.volume_1, axis=1)

	return gx_volume

if __name__ == '__main__':
	gx_volume = get_data()
	print('hola')
