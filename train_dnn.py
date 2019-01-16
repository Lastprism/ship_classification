import os
import pickle
import h5py
import argparse
import time
import glob

import prepare as pp
import config as cfg
import handle_data as hddt
import predict as pd

import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam, SGD
from keras.layers import Input, Dense, Dropout, Flatten, Reshape
from keras.models import Model,load_model,Sequential
import keras
from sklearn import preprocessing
from data_generator import data_generator
from Timer import Timer

def ps(str):
	if cfg.isOut:
		print(str)

def train(x_attributes, y_attributes, train_input, n_leftward_extent, n_rightward_extent):

	###提取输入数据
	#x_attributes = ['mixed_LPS', 'speech_LPS', 'dynamic_noise_LPS', 'LPS_IRM']
	x_train_data = load_train_data('train', x_attributes)
	x_test_data = load_train_data('test', x_attributes)

	###提取输出数据
	#y_attributes = ['speech_LPS', 'LPS_IRM']
	y_train_data = load_train_data('train', y_attributes)
	y_test_data = load_train_data('test', y_attributes)

	###设置训练参数
	batch_size = cfg.batch_size
	epochs = cfg.epochs
	save_times = cfg.save_times
	echo_times = cfg.echo_times
	n_hide = 2048
	

	###建模

	(_, n_concat, n_freq) = x_train_data[0].shape
	model = build_mol_model(n_concat, n_hide, train_input)

	#(_, n_freq) = x_train_data[0].shape
	#model = build_moe_model(n_hide)
	model.summary()

	###训练
	timer = Timer()
	loss = model.evaluate(x_test_data, y_test_data, batch_size = batch_size)

	print("Iteration: %d " % 0, "loss: ", loss)
	last_loss = loss
	save_cnt = 0
	decay_factor = cfg.decay_factor
	decay_rate = cfg.decay_rate
	for iter in range(epochs//echo_times):
		model.fit(x_train_data, y_train_data, batch_size=cfg.batch_size, epochs=save_times, verbose=cfg.verbose)
		loss = model.evaluate(x_test_data, y_test_data, batch_size =  batch_size)
		print("Iteration: %d " % ((iter+1) * save_times), "loss: ", loss)
		if loss < last_loss:
			save_cnt += 1
			if save_cnt % 50 == 0:
				decay_factor *= decay_rate
			last_loss = loss * decay_factor
			save_model(model, (iter+1) * save_times, n_concat, x_attributes)
			ps("save model")
			#pd.test_result(x_attributes, (iter+1) * save_times, n_leftward_extent, n_rightward_extent)

	ps("Training time: %s s" % timer.end())

	###保存模型
	if(not cfg.is_use_last_model):
		save_model(model, epochs, n_concat, x_attributes)


def build_mol_model(n_concat, n_hide, train_input):

	x_input = []
	for item in train_input:
		x_input.append(Input(shape=item[1], name=item[0]))
	x = None
	if(len(train_input) == 1):
		x = x_input[0]
	else:
		x = keras.layers.concatenate(x_input)

	x = Flatten()(x)
	for i in range(6):
		x = Dense(n_hide, activation='sigmoid')(x)
		x = Dropout(0.2)(x)

	y_output = []
	for item in cfg.train_output:
		y_output.append(Dense(item[2], activation=item[1], name=item[0])(x))

	model = Model(inputs=x_input, outputs=y_output)
	model.compile(loss='mean_squared_error',optimizer=Adam(lr=1e-4))
	return model

def build_moe_model(n_hide):

	x_input = []
	for item in cfg.train_input:
		x_input.append(Input(shape=(item[1],), name=item[0]))

	x = keras.layers.concatenate(x_input)
	for i in range(6):
		x = Dense(n_hide, activation='sigmoid')(x)
		x = Dropout(0.2)(x)

	y_output = []
	for item in cfg.train_output:
		y_output.append(Dense(item[2], activation=item[1], name=item[0])(x))

	model = Model(inputs=x_input, outputs=y_output)
	model.compile(loss='mean_squared_error',optimizer=Adam(lr=1e-4))

	return model


def load_train_data(train_type, attributes):
	train_packed_feature_path=os.path.join(cfg.packed_feature_dir, train_type, "%s_data.h5"%train_type)
	train_data = hddt.load_hdf5(train_packed_feature_path, attributes)
	train_data = [train_data[attributes[i]] for i in range(len(attributes))]
	hddt.print_shape(train_data)
	return train_data


def save_model(model, iter, n_concat, x_attributes):
	feature_name = '_'.join(x_attributes)
	model_path = os.path.join(cfg.model_dir, "%d"%(n_concat), feature_name, "Model_%d.h5"%(iter))
	hddt.create_fold(os.path.dirname(model_path))
	model.save(model_path)


