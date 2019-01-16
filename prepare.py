import handle_data as hddt
import os

import config as cfg

def ps(str):
	if cfg.isOut:
		print(str)

#混合语音并写入磁盘
#data = [mixed_complex_x, speech_x, noise_x, mixed_audio_name]

def get_feature(mechant_dir, fishing_dir, data_type):
	des_dir = os.path.join(cfg.feature_dir, data_type)
	hddt.get_feature(mechant_dir, os.path.join(des_dir, 'merchant'), 0)
	hddt.get_feature(fishing_dir, os.path.join(des_dir, 'fishing'), 1)
	ps("get feature finished")

#打包数据
def pack_feature(data_type, n_leftward_extent, n_rightward_extent):
	###读出来特征数据
	feature_dir = os.path.join(cfg.feature_dir, data_type)
	folder_names = os.listdir(feature_dir)
	data = []
	for folder_name in folder_names:
		data += hddt.load_all_feature(os.path.join(feature_dir, folder_name))
	#data = hddt.transpose_matrix(data)
	#[all_mixed_complex_x, all_speech_x, all_noise_x, all_mix_audio_name] = data
	###对特征进行处理并打包，data是一个字典
	data = hddt.process_data(data, n_leftward_extent, n_rightward_extent)
	#computer_scaler(x_all, data_type, snr)
	###将打包后的特征写入磁盘
	hddt.write_packed_feature("%s_data.h5"%data_type, data, data_type)
	ps("pack feature finished!")
