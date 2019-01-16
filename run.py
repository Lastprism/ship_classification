import os
import config as cfg

gpu_id = 1

for i in range(0,4):
	n_leftward_extent = i
	n_rightward_extent = i+1
	get_feature_cmd = "CUDA_VISIBLE_DEVICES=%d python -u test.py get_feature %d %d >> run.log 2>&1"%(gpu_id, i, i+1)
	#get_feature_cmd = "python test.py get_feature %d %d"%(i, i+1)
	os.system(get_feature_cmd)

	for j in range(1, 8):
		train_cmd = "CUDA_VISIBLE_DEVICES=%d python -u test.py train %d %d %d >> run.log 2>&1"%(gpu_id, i, i+1, j)
		#train_cmd = "python test.py train %d %d %d"%(i, i+1, j)
		os.system(train_cmd)
		predict_cmd = "CUDA_VISIBLE_DEVICES=%d python -u test.py predict %d %d %d %d >> run.log 2>&1"%(gpu_id, i, i+1, j, cfg.epochs)
		#predict_cmd = "python test.py predict %d %d %d %d"%(i, i+1, j, cfg.epochs)
		os.system(predict_cmd)