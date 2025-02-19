
"""
@author: Alireza Heshmati
"""

import torch
import numpy as np
import random
import timeit
from os.path import exists
import pickle
import copy
from models import pretrained_model
from dataio import data_preparing
from execution import execute_attack
from utils import print_norm_and_accuracy, plot_results, calculate_result, create_blocks
from arguments import arg


torch.manual_seed(12)
torch.cuda.manual_seed_all(12)
torch.backends.cudnn.deterministic = True

np.random.seed(12)
random.seed(12)

if torch.cuda.device_count() != 0:
    device = torch.device('cuda:0')
else :
    device = torch.device('cpu')
    

# Robust Network over ImageNet Dataset
    
parser = arg()
args, unknown = parser.parse_known_args()
args.model_name = 'Liu2023Comprehensive_Swin-L'
print(f"model name: {args.model_name}")
# best batch 150
net = pretrained_model( args.model_name).to(device)


# load data
args.dataset_type = 'imagenet'
im_net_th= 4
print(f"dataset_type: {args.dataset_type}")
attack_data_test,attack_label_test = data_preparing(net, dataset_type= args.dataset_type,
                                                    device= device, im_net_th= im_net_th)
print(f"Standard accuracy: {len(attack_data_test)/im_net_th}" )

#Untargeted
args.dmax = 0.05

args.size_of_image = 224
args.grouped_stride = 2
args.grouped_len_window = 16

torch.cuda.empty_cache()
torch.manual_seed(12)
torch.cuda.manual_seed_all(12)
torch.backends.cudnn.deterministic = True
np.random.seed(12)
random.seed(12)

last = 4 #args.num_datas
args.attack_batch = 4
args.c = 0.5
args.s_c = 2
args.number_of_modifying = 10
args.learning_rate = 0.2
args.c_l0 = 0.0
args.s_sikma = 0.7
args.c_linf = 0.2

args.grouped_mode = True
args.c_group = 0.15

adv_deltas = torch.zeros_like(attack_data_test[:last])
adv_acc = torch.ones_like(attack_label_test[:last]).numpy()

blocks = create_blocks(stride = args.grouped_len_window ,len_window = args.grouped_len_window,
                          size_of_image = args.size_of_image,number_of_channel = args.number_of_channel)


start = timeit.default_timer()
k=0
for i in range(0, last, args.attack_batch):
    if 0 >= len(adv_acc[i:i+args.attack_batch]):
        break
    k=k+1
    print(k,'.','  batch size :', len(adv_acc[i:i+args.attack_batch]))
    adv_deltas[i:i+args.attack_batch],\
    adv_acc[i:i+args.attack_batch] =execute_attack(net, attack_data_test[:last][i:i+args.attack_batch], 
                                                   attack_label_test[:last][i:i+args.attack_batch] , blocks , args,device) 

stop = timeit.default_timer()
time =stop - start

torch.cuda.empty_cache()



    
if all(adv_acc==0) :
    print('****** No Atack ******')
    result = calculate_result(adv_deltas[:1], adv_acc, time,blocks)
else :
    result = calculate_result(adv_deltas[adv_acc == 1], adv_acc, time,blocks)

if args.group_mode:
    group_mode = 'Grouped'
else: 
    group_mode = 'Simple'

print_norm_and_accuracy(result)

print('   total Time (min) : ', round((time)/60, 3) ,"   time per image (sec) :", round((time) /last, 3)) 

if args.plot :
    np = args.number_of_plots
    outputs = net((attack_data_test[:np]+adv_deltas[:np]).to(device))
    fool_labels = torch.argmax(outputs, 1)
    plot_results(attack_data_test[:np], attack_data_test[:np]+adv_deltas[:np],attack_label_test[:np], fool_labels[:np] ,
                 imagenet_or_cifar10 = args.dataset_type)