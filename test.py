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



magnification = cfg.magnification
n_concat = cfg.n_concat
n_hop = cfg.n_hop
train_snr = cfg.train_snr
test_snr = cfg.test_snr

def test_prepare():
	#混合提取特征
	
	#pp.get_feature(cfg.train_mechant_dir, cfg.train_fishing_dir, 'train')
	pp.get_feature(cfg.test_mechant_dir, cfg.test_fishing_dir, 'test')
	'''
	#处理数据
	pp.pack_feature("train")
	pp.pack_feature("test")
	
	#第一轮训练
	#CUDA_VISIBLE_DEVICES=1
	x_attributes = ['LPS', 'MFCC', 'GFCC']
	y_attributes = ['flag']
	td.train(x_attributes, y_attributes)
	'''
	#最终预测
	pd.predict('fishing')
	pd.predict('merchant')
	
	#评估
	pd.evalue('fishing')
	pd.evalue('merchant')
	

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


def main():
	test_prepare()
	#test_feature()

if __name__ == '__main__':
	main()