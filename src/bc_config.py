
import torch

batch_size = 14
num_workers = 1
epoch = 200
#Binary labels
binary_label_B = 'B'
binary_label_M = 'M'
OP = 'OP'

#Multi labels
multi_label_A = 'A'
multi_label_F = 'F'
multi_label_PT = 'PT'
multi_label_TA = 'TA'
multi_label_DC = 'DC'
multi_label_MC = 'MC'
multi_label_LC = 'LC'
multi_label_PC = 'PC'

binary_label_dict = {binary_label_B: 0, binary_label_M: 1}
multi_label_dict = {multi_label_A:0,multi_label_F:1,multi_label_PT:2,multi_label_TA:3,multi_label_DC:4,multi_label_MC:5,multi_label_LC:6,multi_label_PC:7}

binary_label_list = [binary_label_B,binary_label_M]
multi_label_list = [multi_label_A,multi_label_F,multi_label_PT,multi_label_TA,multi_label_DC,multi_label_MC,multi_label_LC,multi_label_PC]

dataset_path = '/content/drive/MyDrive/matinaMehdizadeh/a_shared_data'


data_path_fold0 = dataset_path + '/Fold_0_5/'
data_path_fold1 = dataset_path + '/Fold_1_5/'
data_path_fold2 = dataset_path + '/Fold_2_5/'
data_path_fold3 = dataset_path + '/Fold_3_5/'
data_path_fold4 = dataset_path + '/Fold_4_5/'

result_path = '/results_bc_5fold/'
tensorboard_path = '/tensorboard_bc_5fold/'

#GPU
gpu0 = torch.device("cuda:0")


#magnification
X40 = '40X'
X100 = '100X' 
X200 = '200X'
X400 = '400X'

#dataset portion
train = 'train'
test = 'test'
val = 'val'

#networks
EfficientNet_b2 = 'EfficientNet_b2'

#Model params
num_classes = 2

#Stain Noramlization
Reinhard_Normalization = 'Reinhard2001'
Macenko_Normalization = 'Macenko2009'
Vahadane_Normalization = 'Vahadane2015'
