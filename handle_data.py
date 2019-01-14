import math
import os
import soundfile
import numpy as np
import argparse
import csv
import time
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import h5py
from sklearn import preprocessing
import librosa
from python_speech_features import mfcc, fbank

import handle_data as hddt
import config as cfg
from gfcc_extractor import gfcc_extractor, cochleagram_extractor

rs = np.random.RandomState(0)

def ps(str):
	if cfg.isOut:
		print(str)

#创建新文件夹
def create_fold(fd):
	if not os.path.exists(fd):
		os.makedirs(fd)
	#os.system("del %s"%os.path.join(fd,*))


#按照路径和采样频率读语音，返回语音和采样频率
def read_audio(path, target_fs = None):
	(audio, fs) = soundfile.read(path)
	if audio.ndim > 1:
		audio = np.mean(audio, axis = 1)
	if target_fs is not None and fs != target_fs:
		audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
		fs = target_fs
	return audio,fs


def write_audio(path, audio, sample_rate):
	soundfile.write(file=path, data=audio, samplerate=sample_rate)


#################################################################################################################

#修整使两段语音一样长
def trim_audio(speech_audio, noise_audio):
	len_speech = len(speech_audio)
	len_noise = len(noise_audio)

	#如果语音长，重复噪音
	if len_noise <= len_speech:
		n_repeat = int(np.ceil(1.0*len(speech_audio) / len(noise_audio)))
		noise_audio = np.tile(noise_audio, n_repeat)[0: len(speech_audio)]
	#如果噪声长，随机取一段和语音一样长数据
	else:
		noise_onset = rs.randint(0, len_noise - len_speech, size=1)[0]
		noise_audio = noise_audio[noise_onset : noise_onset + len_speech]

	return (speech_audio, noise_audio, len(speech_audio))


#计算频谱
def calculate_spectrogram(audio, mode):
	n_window = cfg.n_window
	n_overlap = cfg.n_overlap
	ham_win = np.hamming(n_window)
	[f, t, x] = signal.spectral.spectrogram(audio, window=ham_win, nperseg=n_window, 
		noverlap=n_overlap, detrend=False, return_onesided=True, mode=mode)
	x = x.T
	if mode == 'magnitude':
		x = x.astype(np.float32)
	elif mode == 'complex':
		x = x.astype(np.complex64)
	else:
		raise Exception("calc_sp error mode")
	return x

def trim_feature(feature, frame_cnt):
	if(len(feature) < frame_cnt):
		return np.concatenate([feature, [feature[-1]] * (frame_cnt-len(feature))], axis=0)
	elif(len(feature) > frame_cnt):
		return feature[:frame_cnt]
	else:
		return feature


def get_feature(src_audio_dir, des_feature_dir, flag):
	file_names = [na for na in os.listdir(src_audio_dir) if na.lower().endswith(".wav")]
	target_fs = cfg.fs
	create_fold(des_feature_dir)
	iter = 0
	for file_name in file_names:
		iter += 1
		if(iter%100 == 0):
			ps(iter)
		(audio, _) = read_audio(os.path.join(src_audio_dir, file_name), target_fs)
		data = []
		AS = np.abs(calculate_spectrogram(audio,'complex'))
		tmp = log_sp(AS).astype(np.float32)
		data.append(tmp)
		frame_cnt = data[0].shape[0]

		tmp = mfcc(audio, target_fs, numcep=cfg.n_MFCC, winlen=cfg.n_window/cfg.fs, winstep=cfg.n_overlap/cfg.fs, nfilt=cfg.n_MFCC)
		tmp = trim_feature(tmp, frame_cnt)
		data.append(tmp)

		cochlea = cochleagram_extractor(audio, target_fs, cfg.n_window, cfg.n_overlap, cfg.n_GFCC_IRM, 'hamming')
		tmp = trim_feature(gfcc_extractor(cochlea, cfg.n_GFCC_IRM, cfg.n_GFCC).T, frame_cnt)
		data.append(tmp)

		data.append(flag)
		file_name = '%s.%s'%(file_name.split('.')[0],file_name.split('.')[1])
		pickle.dump(data, open(os.path.join(des_feature_dir, "%s.p"%file_name), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


#################################################################################################################################

def transpose_matrix(matrix):
	return list(map(list, zip(*matrix)))

def load_all_feature(feature_dir):
	data = []
	names = os.listdir(feature_dir)
	for name in names:
		feature_path = os.path.join(feature_dir, name)
		data.append(pickle.load(open(feature_path, 'rb')))
	return data

def print_shape(data):
	if cfg.isOut:
		if type(data) == type(dict()):
			for key in data:
				print(key, " : ", data[key].shape)
			print('\n')
		elif type(data) == type([]):
			for item in data:
				print(item.shape)
			print('\n')

def log_sp(x):
	return np.log(x + 1e-8)

def x_transpose_2d_to_3d(mixed_x):
	#获取mixed_x的shape
	(x,y) = mixed_x.shape
	#对mixed_x的前后进行补全0行
	n_leftward_extent = max(cfg.n_leftward_extent, 1)
	tmp_x = np.vstack((np.array([mixed_x[0]] * n_leftward_extent) , mixed_x, np.array([mixed_x[-1]] * cfg.n_rightward_extent)))
	#存储结果
	res = []
	#从每一帧的前后几帧组合起来 + 静态噪音LPS
	for iter in range(n_leftward_extent, x + n_leftward_extent):
		res.append( np.vstack( ( tmp_x[ iter-cfg.n_leftward_extent : iter+cfg.n_rightward_extent ] ) ) )
	#返回结果
	return np.array(res)


def process_data(data):
	res = dict({'LPS':[], 'MFCC':[], 'GFCC':[], 'flag':[]})
	iter = 0
	for item in data:
		if iter % 100 == 0:
			ps(iter)
		iter += 1
		LPS = np.array(item[0])
		MFCC = np.array(item[1])
		GFCC = np.array(item[2])
		flag = None
		length = len(LPS)

		if(item[3] == 0):
			flag = np.zeros((length,1))
		else:
			flag = np.ones((length,1))

		######################################
		LPS = x_transpose_2d_to_3d(LPS)
		MFCC = x_transpose_2d_to_3d(MFCC)
		GFCC = x_transpose_2d_to_3d(GFCC)
		res['LPS'].append(LPS)
		res['MFCC'].append(MFCC)
		res['GFCC'].append(GFCC)
		res['flag'].append(flag)

		######################################

	for key in res:
		res[key] = np.concatenate(res[key], axis=0)
	
	print_shape(res)
	return res


def write_scaler(scaler, data_type, snr):
	packed_feature_dir = os.path.join(cfg.packed_feature_dir, "spectrogram", data_type, "%ddb"%snr)
	create_fold(packed_feature_dir)
	packed_feature_path = os.path.join(packed_feature_dir, "scaler.p")
	pickle.dump(scaler, open(packed_feature_path, 'wb'))


def computer_scaler(x_all, data_type, snr):
	x = np.array(x_all)
	(n_segs, n_concat, n_freq) = x.shape
	x2d = x.reshape((n_segs*n_concat, n_freq))
	scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(x2d)
	write_scaler(scaler, data_type, snr)


def write_packed_feature(filename, data, data_type):
	packed_feature_dir = os.path.join(cfg.packed_feature_dir, data_type)
	create_fold(packed_feature_dir)
	#dir_len = len(os.listdir(packed_feature_dir))
	packed_feature_path = os.path.join(packed_feature_dir,filename)
	with h5py.File(packed_feature_path, 'w') as hf:
		for key in data:
			hf.create_dataset(key, data=data[key])


############################################
def load_hdf5(packed_feature_path, attributes):
	res = dict()
	with h5py.File(packed_feature_path, 'r') as hf:
		for attr in attributes:
			x = hf.get(attr)
			x = np.array(x)
			res[attr] = x
	return res

##############################################################


def np_mean_square_error(predict_all, y_all):
	loss = 0
	for i in range(len(y_all)):
		loss += np.mean(np.square(predict_all[i] - y_all[i]))
	return loss


def eval(model ,generator, x, y):

	predict_all, y_all = [[] for i in range(len(y))], [[] for i in range(len(y))]
	for (batch_x, batch_y) in generator.generate(xs=x,ys=y):
		pred = model.predict(batch_x)

		for i in range(len(y)):
			predict_all[i].append(pred[i])
			y_all[i].append(batch_y[i])

	for i in range(len(y)):
		predict_all[i] = np.concatenate(predict_all[i], axis=0)
		y_all[i] = np.concatenate(y_all[i], axis=0)

	return np_mean_square_error(predict_all, y_all)