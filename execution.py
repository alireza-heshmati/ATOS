"""
@author: Alireza Heshmati
"""

import torch
from utils import creat_targets
from attack_utils import original_attack, original_attack_GPU
import copy
    
def execute_attack(model, images, labels,blocks,args, device):

    images = images.to(device)
    labels = labels.to(device)
    
    if args.dataset_type == 'cifar10' :
        num_class = 10
    else :
        num_class = 1000 
    
    # initialization 
    outputs = model(images) 
    one_hot_labels = torch.eye(num_class)[labels.to(torch.device('cpu'))].to(device)
    targets = creat_targets(outputs, one_hot_labels, args.target_type,device)
    
    adv_deltas = torch.zeros_like(images)
    adv_acc = torch.zeros_like(labels).float().cpu().detach()
    
    c = copy.deepcopy(args.c) 
    ind = adv_acc == 0

    if args.o_f_type == 'CE':
        for k in range(args.number_of_modifying):
            torch.cuda.empty_cache()
            if targets != None:
                targets_ = targets[ind]
            else :
                targets_ = None

            if args.total_gpu:
                adv_deltas[ind], adv_acc[ind] = original_attack_GPU( model,
                                                                    images[ind],
                                                                    adv_deltas[ind],
                                                                    one_hot_labels[ind],
                                                                    targets_,
                                                                    args, device,
                                                                    blocks)
            else:
                adv_deltas[ind], adv_acc[ind] = original_attack( model,
                                                                images[ind],
                                                                adv_deltas[ind],
                                                                one_hot_labels[ind],
                                                                targets_,
                                                                args, device,
                                                                blocks)
            ind = adv_acc == 0
            print('CE loss,','  c  :',args.c ,'  c_l0  :',args.c_l0 ,'  c_group  :',args.c_group ,'  c_linf  :',args.c_linf ,
                     '  number of attacked :',len(adv_acc[adv_acc == 1]))
            if all(~ind):
                break

            args.c = round(args.c * args.s_c,4)
            
    elif args.o_f_type == 'CW':
        kap = copy.deepcopy(args.kap) 
        for k in range(args.number_of_modifying):
            for _ in range(1):
                if targets != None:
                    targets_ = targets[ind]
                else :
                    targets_ = None

                if args.total_gpu:

                    adv_deltas[ind], adv_acc[ind] = original_attack_GPU( model,
                                                                        images[ind],
                                                                        adv_deltas[ind],
                                                                        one_hot_labels[ind],
                                                                        targets_,
                                                                        args, device,
                                                                        blocks)
                else:
                    adv_deltas[ind], adv_acc[ind] = original_attack( model,
                                                                    images[ind],
                                                                    adv_deltas[ind],
                                                                    one_hot_labels[ind],
                                                                    targets_,
                                                                    args, device,
                                                                    blocks)
                
                ind = adv_acc == 0
                print('CW loss,','  c  :',args.c ,'  c_l0  :',args.c_l0 ,'  c_group  :',args.c_group,'  c_linf  :',args.c_linf , '  kap  :',args.kap , '  number of attacked :',len(adv_acc[adv_acc == 1]))
                if all(~ind):
                    break
                if args.kap == 0:
                    args.kap =1
                else :
                    args.kap = args.kap * args.kap_s
            args.kap = kap
            
            if all(~ind):
                break

            args.c = round(args.c * args.s_c,4)
    args.c = c
    return adv_deltas.detach().cpu(), adv_acc.detach().cpu().numpy()
