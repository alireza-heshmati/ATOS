
"""
@author: Alireza Heshmati
"""

import torch
from utils import objective_f



def LSEAp(delx,p):
    max_= torch.amax(abs(delx), dim =(-3,-2,-1),keepdim=True)
    e = torch.exp(p*(abs(delx) - max_))
    loss = torch.log(torch.sum(e,dim =(-3,-2,-1))) / p
    return torch.sum(loss)

def LSEAp_grad(delx,p):
    max_= torch.amax(abs(delx), dim =(-3,-2,-1),keepdim=True).expand_as(delx)
    e = torch.exp(p*(abs(delx) - max_))
    sign = torch.sign(delx)
    sum_ = torch.sum(e,dim =(-3,-2,-1),keepdim=True).expand_as(delx)
    return sign * e / sum_

def GSl0(delx, sikma2, type_ ='RGB'):
    if type_ == 'RGB':
        exp_part = torch.exp(-torch.sum(delx**2, dim = 1)/(2*sikma2))
    else :
        exp_part = torch.exp(-(delx**2)/(2*sikma2))
    return - torch.sum(exp_part)

def GSl0_grad(delx,sikma2, type_ ='RGB'):
    if type_ == 'RGB':
        exp_part = torch.exp(-torch.sum(delx**2, dim = 1,keepdim=True)/(2*sikma2)).expand_as(delx)
    else :
        exp_part = torch.exp(-(delx**2)/(2*sikma2))
    return delx * exp_part / sikma2

def OSL0(delx, sikma2, kernel_size, stride):
    delta_groups = delx.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    exp_part = torch.exp(-torch.sum(delta_groups **2, dim = (1,-2,-1))/(2*sikma2))
    return  - torch.sum(exp_part)


def OSL0_grad(delx,sikma2,Blocks):
    delta_B = torch.mul(Blocks, delx)
    exp_part = torch.exp(-torch.sum(delta_B**2, dim = (-3,-2,-1),keepdim=True)/(2*sikma2)).expand_as(delta_B)
    return torch.sum(delta_B * exp_part, dim =0) / sikma2


def define_first_sikma2_convex(x2_max, d, stride):
    if d == stride:
        sikma2 = d * x2_max
        
    else:
        sikma2 = (2*d-1) * x2_max
        
    return sikma2


def define_mu(sikma2, x2_max, d, stride):
    return sikma2

def create_mu_and_sikma2(x2_max = 1, d = 2, stride = 2, s_sikma = 0.5,  n_sikma = 10):
    sikma2_list = []
    mu_list = []
    
    sikma2_list.append(define_first_sikma2_convex(x2_max, d, stride))
    mu_list.append(define_mu(sikma2_list[-1], x2_max, d, stride) )
    
    for i in range(1,n_sikma):
        sikma2_list.append(sikma2_list[-1]*s_sikma**2)
        mu_list.append(define_mu(sikma2_list[-1], x2_max, d, stride))
    
    return sikma2_list, mu_list



def original_attack_GPU( model, images, delta, one_hot_labels,
                     targets, args, device, blocks = None):
    
    model.eval()
    
    labels = torch.argmax(one_hot_labels , 1)
    
    if args.sparse_mode == "RGB":
        d = 3
        stride = 3
    else:
        d = 1
        stride = 1
        
        
    sikma2_list, mu_list = create_mu_and_sikma2(x2_max = args.dmax **2 , 
                                                d = d, stride = stride,
                                                 s_sikma = args.s_sikma,
                                                  n_sikma = args.outer_iter)
    
    # prepare blocks
    if args.c_group != 0 and blocks != None:
        sikma2_list_b, mu_list_b = create_mu_and_sikma2(x2_max = args.dmax **2 , 
                                                        d = 3 * args.grouped_len_window,
                                                        stride = args.grouped_stride, 
                                                        s_sikma = args.s_sikma, 
                                                        n_sikma = args.outer_iter)
    
    # algorithm
    for step in range(args.outer_iter):
        
        for i in range(args.inner_iter) :

            delta.requires_grad_(True)

            model_outputs = model((images+delta))
            
            total_loss = args.c* objective_f(model_outputs, 
                                             one_hot_labels, 
                                             targets, device,
                                             args.o_f_type, 
                                             args.kap)
            
            if args.c_group != 0 and blocks != None:
                total_loss += args.c_group * mu_list_b[step] * OSL0(delta,
                                                                        sikma2_list_b[step],
                                                                        args.grouped_len_window,
                                                                        args.grouped_stride)
            if args.c_l0 != 0:
                total_loss += args.c_l0 * mu_list[step] *\
                        GSl0(delta, sikma2_list[step], args.sparse_mode)
                
            if args.c_linf != 0:
                total_loss += args.c_linf * LSEAp(delta,args.p)
                
            total_loss.backward()

            with torch.no_grad():
                gradiant = delta.grad 
                delta = delta - args.learning_rate * gradiant
                delta = torch.clamp( images+delta , min = 0 , max = 1) - images
                
                # optional
                if args.eta_linf < 1:
                    delta = torch.clamp( delta , min = -args.eta_linf, max = args.eta_linf)
                    
            if torch.all(gradiant == 0) :
                break

    if args.is_pixel_scale:
        delta = torch.round((delta)*255)/255
    
    if targets == None:
        adv_acc = (labels != torch.argmax(model(images+delta), 1)).float()
    else :
        adv_acc = (targets == torch.argmax(model(images+delta), 1)).float()
        
    return delta, adv_acc.cpu().detach()




def original_attack( model, images, delta, one_hot_labels,
                     targets, args, device, blocks = None):
    
    model.eval()
    
    labels = torch.argmax(one_hot_labels , 1)
    
    if args.sparse_mode == "RGB":
        d = 3
        stride = 3
    else:
        d = 1
        stride = 1
        
        
    sikma2_list, mu_list = create_mu_and_sikma2(x2_max = args.dmax **2 , 
                                                d = d, stride = stride,
                                                 s_sikma = args.s_sikma,
                                                  n_sikma = args.outer_iter)
    
    # prepare blocks
    if args.c_group != 0 and blocks != None and args.OSL0_grad_exec != "torch":
        batch_num  = len(images)
        block_num = len(blocks)
        v = torch.ones((batch_num,block_num,3,images.shape[-2],images.shape[-1]))
        B = torch.mul(v, blocks).to(device)
        B = B.permute((1,0,2,3,4))
        
    if args.c_group != 0 and blocks != None:
        sikma2_list_b, mu_list_b = create_mu_and_sikma2(x2_max = args.dmax **2 , 
                                                        d = 3 * args.grouped_len_window,
                                                        stride = args.grouped_stride, 
                                                        s_sikma = args.s_sikma, 
                                                        n_sikma = args.outer_iter)
    
    # algorithm
    for step in range(args.outer_iter):
        
        for i in range(args.inner_iter) :

            delta.requires_grad_(True)

            model_outputs = model((images+delta))

            objective_function = objective_f(model_outputs, 
                                             one_hot_labels, 
                                             targets, device,
                                             args.o_f_type, 
                                             args.kap)
            
            if args.c_group != 0 and blocks != None and args.OSL0_grad_exec == "torch":
                group_loss = OSL0(delta,
                                    sikma2_list_b[step],
                                    args.grouped_len_window,
                                    args.grouped_stride)
                
                torch_loss = args.c_group* mu_list_b[step] * group_loss + args.c*objective_function
            else :
                torch_loss = args.c* objective_function

            torch_loss.backward()
            
            with torch.no_grad():
                gradiant = delta.grad 
                
                if args.c_l0 != 0:
                    gradiant += args.c_l0 * mu_list[step] *\
                          GSl0_grad(delta, sikma2_list[step], args.sparse_mode)
                    
                if args.c_linf != 0:
                    
                    gradiant += args.c_linf * LSEAp_grad(delta,args.p)
                    
                if args.c_group != 0 and blocks != None and args.OSL0_grad_exec != "torch":
                    gradiant += args.c_group* mu_list_b[step] *\
                        OSL0_grad(delta,sikma2_list_b[step],B)
                    
                delta = delta - args.learning_rate * gradiant
                delta = torch.clamp( images+delta , min = 0 , max = 1) - images
                
                # optional
                if args.eta_linf < 1:
                    delta = torch.clamp( delta , min = -args.eta_linf, max = args.eta_linf)
                    
            if torch.all(gradiant == 0) :
                break
            
    if args.is_pixel_scale:
        delta = torch.round((delta)*255)/255
    
    if targets == None:
        adv_acc = (labels != torch.argmax(model(images+delta), 1)).float()
    else :
        adv_acc = (targets == torch.argmax(model(images+delta), 1)).float()
    
    return delta, adv_acc.cpu().detach()

