import numpy as np
import os
import pickle
import h5py
import argparse
import time
import glob
import matplotlib.pyplot as plt
import json

import prepare as pp
import config as cfg
import handle_data as hddt
from spectrogram_to_wave import recover_wav

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.models import load_model

def ps(str):
	if cfg.isOut:
		print(str)

def load_Model(Iter):
	model_path = os.path.join(cfg.model_dir,"Model_%d.h5"%(Iter))
	return load_model(model_path)

def write_json(file_name, data):
	with open(file_name, 'w') as f:
		json.dump(data, f, indent=4)

def read_json(file_name):
	with open(file_name, 'r') as f:
		return json.load(f)

def predict(ship_type):

	###加载模型
	mol_model = load_Model(cfg.test_iter)
	###加载数据
	feature_dir = os.path.join(cfg.feature_dir, "test", ship_type)
	feature_names = os.listdir(feature_dir)
	res = dict()
	iter = 0
	for f_name in feature_names:
		iter += 1
		if iter % 100 == 0:
			ps(iter)
		###读取特征
		data = pickle.load(open(os.path.join(feature_dir, f_name), 'rb'))
		[LPS, MFCC, GFCC, flag] = pickle.load(open(os.path.join(feature_dir, f_name), 'rb'))

		LPS = hddt.x_transpose_2d_to_3d(LPS)
		MFCC = hddt.x_transpose_2d_to_3d(MFCC)
		GFCC = hddt.x_transpose_2d_to_3d(GFCC)

		###数据处理

		###mol模型预测
		tmp_res = mol_model.predict([LPS, MFCC, GFCC])

		zero_cnt = 0
		one_cnt = 0
		for x in tmp_res:
			if x > 0.5:
				one_cnt += 1
			else:
				zero_cnt += 1
		if zero_cnt < one_cnt:
			res[f_name] = 1
		else:
			res[f_name] = 0
		print(ship_type, zero_cnt, one_cnt)
	res_path = os.path.join(cfg.res_dir, "%s.txt"%ship_type)
	hddt.create_fold(os.path.dirname(res_path))
	write_json(res_path, res)

def evalue(ship_type):
	file_name = os.path.join(cfg.res_dir, "%s.txt"%ship_type)
	data = read_json(file_name)
	true_cnt = 0
	false_cnt = 0
	for key in data:
		if data[key] == cfg.ans[ship_type]:
			true_cnt += 1
		else:
			false_cnt += 1
	ps("%s  True rate:%.2f%%   False rate:%.2f%%"%(ship_type, true_cnt*100.0/(true_cnt+false_cnt), false_cnt*100.0/(true_cnt+false_cnt)))