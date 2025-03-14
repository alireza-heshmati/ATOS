
"""
@author: Alireza Heshmati
"""

import numpy as np
from six.moves import cPickle as pickle
import torch
import pandas as pd
import os
from PIL import Image
import torchvision.transforms as transforms

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding ='latin1')
    fo.close()
    return dict

def data_preparing(net, dataset_type='cifar10', device = torch.device('cuda:0'), im_net_th = 3000):
    torch.cuda.empty_cache()
    net.eval()
    if dataset_type =='cifar10':
        batch = 64
        rtest = unpickle(
            "YOUR_PATH_OF_CIFAR10")
        test_label = torch.Tensor(np.array(rtest['labels'],dtype=np.float32)).type(torch.LongTensor) 
        test_data = torch.Tensor(np.array(rtest['data'].reshape(rtest['data'].shape[0],3,32,32),dtype=np.float32))/255.0
         
        
    if dataset_type =='imagenet':
        batch = 32
        val_mapping = pd.read_csv(
            './supplies/LOC_synset_mapping.csv')  
        val_mapping_dict=val_mapping.to_dict()
        values_keys_mapping = {v:k for k,v in val_mapping_dict['labelName'].items()}
        val_solution = pd.read_csv(
            './supplies/LOC_val_solution.csv')  
        val_solution_dict=val_solution.to_dict()
        values_keys = {v:k for k,v in val_solution_dict['ImageId'].items()}
           #create dataset 
        test_data = []
        test_label = []
        dirs = os.listdir( 
            './supplies/ILSVRC_img_val_samples/' )
        k=0
        for i in dirs:
            img = Image.open(
                './supplies/ILSVRC_img_val_samples/'+i)
            img_tensor = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()])(img)
            if img_tensor.shape[0] == 3 :
                k+=1
                test_data.append(img_tensor)
                name_of_image = i.split('.')[0]
                code_of_label = val_solution_dict['PredictionString'][values_keys[name_of_image]].split(' ')[0]
                label = values_keys_mapping[code_of_label]
                test_label.append(torch.tensor(label))
            if len(test_label) == im_net_th:
                break
        test_data = torch.stack(test_data)
        test_label = torch.stack(test_label)    
        
    att_data = []
    att_label = []
    for i in range(0, len(test_data), batch): # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
        batch_X = test_data[i:i+batch]
        batch_y = test_label[i:i+batch]  
        
        with torch.no_grad():
            labelpredict =net(batch_X.to(device)).cpu()
        pre = torch.argmax(labelpredict, 1)
        ind = pre == batch_y
        att_data.append(batch_X[ind])
        att_label.append(batch_y[ind])

    attack_data_test=[]
    attack_test_label=[]
    for k in range(len(att_data)):
        for j in range(len(att_data[k])):
            attack_data_test.append(att_data[k][j])
            attack_test_label.append(att_label[k][j])  
    attack_data_test=torch.stack(attack_data_test)
    attack_test_label = torch.stack(attack_test_label)
    
    return attack_data_test,attack_test_label
    
    
    
    
    
    
    
    
