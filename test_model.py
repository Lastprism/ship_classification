def data_generator(tr_x1, tr_x2, batch_size = 32, epochs = 100):
	rs = np.random.RandomState(0)
	rs.shuffle(tr_x1)
	rs.shuffle(tr_x2)

	data_len = len(tr_x1)

	while True:
		x1 = rs.choice(tr_x1, batch_size, replace=False).reshape(-1,1)
		x2 = rs.choice(tr_x2, batch_size, replace=False).reshape(-1,1)
		y = x1 * x2
		yield x1, x2, y

def build_model(flag):
	if flag == 1 :
		return load_model('test_mod.h5')
	else:
		x1 = Input(shape=(1,), dtype='float32', name='x1')
		x2 = Input(shape=(1,), dtype='float32', name='x2')
		x = keras.layers.concatenate([x1, x2])
		x = Dense(300, activation='sigmoid')(x)
		x = Dense(300, activation='sigmoid')(x)
		x = Dense(300, activation='sigmoid')(x)
		x = Dense(300, activation='sigmoid')(x)
		y = Dense(1, activation='linear', name='y')(x)

		model=Model(inputs=[x1, x2],outputs=[y])

		model.compile(optimizer='Adam', loss='mean_squared_error')
		return model

def train_model(flag, tr_x1, tr_x2, model):
	if flag == 1:
		return model

	if flag == 2:
		cnt = 0
		batch_size = 32
		for x1,x2,y in data_generator(tr_x1, tr_x2, batch_size):
			cnt += 1
			model.train_on_batch([x1,x2],[y])

			if(cnt % 1000 == 0):
				print(cnt, " : " ,model.evaluate([x1, x2], [y], batch_size, 0))

			if(cnt == 20000):
				break
		return model


def test_Model():
	tr_x1 = [x for x in range(-100,100)]
	tr_x2 = [x for x in range(-100,100)]

	model = build_model(1)
	
	#model.summary()

	model = train_model(1, tr_x1, tr_x2, model)

	#model.fit([tr_x1,tr_x1],[tr_y1, tr_y2, tr_y3],epochs=100, batch_size=100)

	dg = data_generator(tr_x1, tr_x2, 5)
	x1, x2, y = dg.__next__()
	print(y)
	res = model.predict([x1, x2])

	print(np.rint(res))
	model.save('test_mod.h5')
	
