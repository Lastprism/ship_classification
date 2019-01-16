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

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.models import load_model

def ps(str):
	if cfg.isOut:
		print(str)

def load_Model(x_attributes, n_concat, Iter):
	model_path = ""
	if(cfg.is_use_last_model):
		model_dir = os.path.join(cfg.model_dir, "%d"%n_concat, '_'.join(x_attributes))
		model_name = os.listdir(model_dir)
		model_path = os.path.join(model_dir, model_name[-1])
	else:
		model_path = os.path.join(cfg.model_dir, "%d"%n_concat, '_'.join(x_attributes), "Model_%d.h5"%(Iter))
	return load_model(model_path)

def write_json(file_name, data):
	with open(file_name, 'w') as f:
		json.dump(data, f, indent=4)

def read_json(file_name):
	if(not os.path.exists(file_name)):
		dic = dict()
		write_json(file_name, dic)
	with open(file_name, 'r') as f:
		return json.load(f)

def predict(x_attributes, ship_type, iter, n_leftward_extent, n_rightward_extent, data_type="test"):

	###加载模型
	n_concat = n_leftward_extent + n_rightward_extent
	mol_model = load_Model(x_attributes, n_concat, iter)
	###加载数据
	feature_dir = os.path.join(cfg.feature_dir, data_type, ship_type)
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

		LPS = hddt.x_transpose_2d_to_3d(LPS, n_leftward_extent, n_rightward_extent)
		MFCC = hddt.x_transpose_2d_to_3d(MFCC, n_leftward_extent, n_rightward_extent)
		GFCC = hddt.x_transpose_2d_to_3d(GFCC, n_leftward_extent, n_rightward_extent)

		dic = dict({'LPS':LPS, 'MFCC':MFCC, 'GFCC':GFCC})
		x_input = []
		for key in x_attributes:
			x_input.append(dic[key])

		###数据处理

		###mol模型预测
		tmp_res = mol_model.predict(x_input)

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
		#print(ship_type, zero_cnt, one_cnt)
	res_path = os.path.join(cfg.res_dir, "%s_%s.txt"%(ship_type,data_type))
	hddt.create_fold(os.path.dirname(res_path))
	write_json(res_path, res)

def evalue(x_attributes, ship_type, n_concat, data_type):
	file_name = os.path.join(cfg.res_dir, "%s_%s.txt"%(ship_type,data_type))
	data = read_json(file_name)
	true_cnt = 0
	false_cnt = 0
	for key in data:
		if data[key] == cfg.ans[ship_type]:
			true_cnt += 1
		else:
			false_cnt += 1
	result_name = os.path.join(cfg.res_dir,"result.txt")
	res = read_json(result_name)
	key = "%d_%s_%s_%s"%(n_concat, '_'.join(x_attributes), data_type, ship_type)
	res[key] = true_cnt*100.0/(true_cnt+false_cnt)
	write_json(result_name, res)
	ps("%s %s  True rate:%.2f%%   False rate:%.2f%%"%(data_type, ship_type, true_cnt*100.0/(true_cnt+false_cnt), false_cnt*100.0/(true_cnt+false_cnt)))

def test_result(x_attributes, iter, n_leftward_extent, n_rightward_extent, data_type):
	n_concat = n_leftward_extent + n_rightward_extent
	#最终预测
	predict(x_attributes, 'fishing', iter, n_leftward_extent, n_rightward_extent, data_type)
	predict(x_attributes, 'merchant', iter, n_leftward_extent, n_rightward_extent, data_type)
	
	#评估
	evalue(x_attributes, 'fishing', n_concat, data_type)
	evalue(x_attributes, 'merchant', n_concat, data_type)