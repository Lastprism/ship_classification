import handle_data as hddt
import prepare as pp
import train_dnn as td
import predict as pd
from gfcc_extractor import gfcc_extractor, cochleagram_extractor
import config as cfg
import os
import json
from keras.layers import Input, Dense, Dropout
from keras.models import Model,load_model
import keras
from keras.optimizers import Adam
from sklearn import preprocessing
import numpy as np
from Timer import Timer
import evaluate as eva
import librosa
from python_speech_features import mfcc, fbank
import sys



def get_feature(n_leftward_extent, n_rightward_extent):
	#混合提取特征
	
	pp.get_feature(cfg.train_mechant_dir, cfg.train_fishing_dir, 'train')
	pp.get_feature(cfg.test_mechant_dir, cfg.test_fishing_dir, 'test')
	
	#处理数据
	pp.pack_feature("train", n_leftward_extent, n_rightward_extent)
	pp.pack_feature("test", n_leftward_extent, n_rightward_extent)
	
def train(train_input, n_leftward_extent, n_rightward_extent):
	#CUDA_VISIBLE_DEVICES=1
	x_attributes = [ item[0] for item in train_input ]
	y_attributes = ['flag']
	td.train(x_attributes, y_attributes, train_input, n_leftward_extent, n_rightward_extent)

def predict(train_input, iter, n_leftward_extent, n_rightward_extent):
	#最终预测
	x_attributes = [ item[0] for item in train_input ]
	pd.test_result(x_attributes, iter, n_leftward_extent, n_rightward_extent, 'train')
	pd.test_result(x_attributes, iter, n_leftward_extent, n_rightward_extent, 'test')


def get_input(j, n_concat):
	train_input = []
	if j&1:
		x = ['LPS',(n_concat, cfg.n_freq)]
		train_input.append(x)
	if j&2:
		x = ['MFCC',(n_concat, cfg.n_MFCC)]
		train_input.append(x)
	if j&4:
		x = ['GFCC',(n_concat, cfg.n_GFCC)]
		train_input.append(x)
	return train_input


if __name__ == '__main__':
	parameter = []
	parameter.append(sys.argv[1])
	for i in range(2,len(sys.argv)):
		parameter.append(eval(sys.argv[i]))

	n_leftward_extent = parameter[1]
	n_rightward_extent = parameter[2]
	n_concat = n_leftward_extent + n_rightward_extent
	

	if(parameter[0] == 'get_feature'):
		get_feature(n_leftward_extent, n_rightward_extent)
	elif(parameter[0] == 'train'):
		train_input = get_input(parameter[3], n_concat)
		train(train_input, n_leftward_extent, n_rightward_extent)
	elif(parameter[0] == 'predict'):
		train_input = get_input(parameter[3], n_concat)
		iter = parameter[4]
		predict(train_input, iter, n_leftward_extent, n_rightward_extent)


def test_feature():
	filename = 'test.wav'
	(audio, _) = hddt.read_audio(filename, cfg.fs)
	as_feature = hddt.calculate_spectrogram(audio, 'magnitude')


	cochlea = cochleagram_extractor(audio, cfg.fs, 512, 256, 64, 'hamming')
	gfcc_feature = gfcc_extractor(cochlea, 64, 30).T


	mfcc_feature = mfcc(audio, cfg.fs, numcep=40, winlen=512/cfg.fs, winstep=256/cfg.fs, nfilt=40)
	mel_power = fbank(audio, cfg.fs, winlen=512/cfg.fs, winstep=256/cfg.fs, nfilt=40)
	#d_mfcc_feat = delta(mfcc_feat, 2)
	#fbank_feat = logfbank(sig,rate)

	print(audio.shape)
	print(as_feature.shape)
	print(gfcc_feature.shape)
	print(mfcc_feature.shape)
	print(mel_power[0].shape)
	print(mel_power[1].shape)
	print(np.sum(mel_power[0][0]))
	print(mel_power[1][0])
