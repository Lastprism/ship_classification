import os


fs = 16000
n_window = 512      # windows size for FFT
n_overlap = 256     # overlap of window

frame_count = 6		
n_rightward_extent = 2
n_leftward_extent = 1
n_forward_hop = 1
n_concat = n_rightward_extent + n_leftward_extent


###global parameter
batch_size = 256
epochs = 1000
echo_times = 1
save_times = 1
train_snr = 0
test_snr = 0
magnification = 2
n_concat = 7
n_hop = 1
isOut = True

n_MFCC = 41
n_MFCC_IRM = 40
n_GFCC = 30
n_GFCC_IRM = 64


###path parameter
workspace = os.path.join("..", "Data", "ship_classification", "workspace")
feature_dir = os.path.join(workspace, "feature")
packed_feature_dir = os.path.join(workspace, "packed_feature")
model_dir = os.path.join(workspace, "model", "v1_3")
res_dir = os.path.join(workspace, "result")


mini_data_dir = os.path.join("..", "Data", "ship_classification", "mini_data")
train_mechant_dir = os.path.join(mini_data_dir, "train", "merchant_ship")
train_fishing_dir = os.path.join(mini_data_dir, "train", "fishing_ship")
test_mechant_dir = os.path.join(mini_data_dir, "test", "merchant_ship")
test_fishing_dir = os.path.join(mini_data_dir, "test", "fishing_ship")



###网络配置
n_freq = 257
n_concat = n_leftward_extent + n_rightward_extent


train_input = [['LPS', (n_concat, n_freq)], ['MFCC',(n_concat, n_MFCC)], ['GFCC',(n_concat, n_GFCC)]]
train_output = [['flag', 'sigmoid', 1]]


test_iter = 22

ans = dict({'fishing':1, 'merchant':0})

verbose = 1

decay_factor = 1.0
decay_rate = 0.95