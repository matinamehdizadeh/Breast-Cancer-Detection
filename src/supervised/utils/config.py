import numpy as np

image_size = 224
image_mean = [0.4914, 0.4822, 0.4465]
image_std = [0.2023, 0.1994, 0.2010]
batch_size = 8 #for test, otherwise 7
num_workers = 1

gpu0 = 'cuda:0'

result_path = '~/results_bc/'
tensorboard_path = '~/tensorboard_bc/'

