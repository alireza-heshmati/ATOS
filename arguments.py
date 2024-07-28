import argparse

def arg():
    '''
    input hyper parameters
    '''
    parser = argparse.ArgumentParser(description='EW,_PW_and_GW_Attacks')
    
    #general setting
    
    parser.add_argument('-f')
    
    parser.add_argument('--model_name', type=str, default='Liu2023Comprehensive_Swin-L',
                        help='name of pretrained robust model Liu2023Comprehensive_Swin-L for imagenet')
    
    parser.add_argument('--is_pixel_scale', type=bool, default=True,
                        help='scale the examples values in 0 to 1 according to 0 to 255.')
    
    parser.add_argument('--total_gpu', type=bool, default=False,
                        help='wheter use gpu to compute all losses or not, False and True')
    
    parser.add_argument('--dataset_type', type=str, default='imagenet',
                        help='base dataset, just cifar10 and imagenet')
    
    parser.add_argument('--target_type', type=str, default='Untargeted',
                        help='kind of target or untargeted attack, just Untargeted,\
                              Best, Average, Worst')
    
    parser.add_argument('--o_f_type', type=str, default='CE',
                        help='objective function type, just CE (Cross Entropy) and CW')
    
    parser.add_argument('--kap', type= float, default= 0,
                        help='confidence in CW objective function')
    
    parser.add_argument('--kap_s', type= float, default= 2,
                        help='scalse of confidence in CW objective function')
    
    parser.add_argument('--num_datas', type= int, default=1024,
                        help='number of datas for attack, for cifar10 : 1024 and\
                              for imagenet : 1024')
    
    parser.add_argument('--number_of_modifying', type= float, default= 5,
                        help='number of modifying mode')
    
    parser.add_argument('--attack_batch', type= int, default= 256,
                        help='batch size under each attack')
    
    parser.add_argument('--size_of_image', type= int, default= 32,
                        help='size of input image, 32 for cifar10 and 224 for imagenet ')
    
    parser.add_argument('--number_of_channel', type= int, default= 3,
                        help='number of input channel, just 3')
    
    parser.add_argument('--plot', type= bool, default= True,
                        help='ploting some outputs of attack and originals')
    
    parser.add_argument('--number_of_plots', type= int, default= 8,
                        help='number of images for ploting')
    
    
    
    # Original attack setting
    
    parser.add_argument('--grouped_mode', type= bool, default= False,
                        help='grouped mode for attack')
    
    parser.add_argument('--grouped_stride', type= int, default= 2,
                        help='stride of splitting of grouped window')
    
    parser.add_argument('--grouped_len_window', type= int, default= 16,
                        help='length of grouped window')
    
    parser.add_argument('--OSl0_grad_exec', type= str, default= "manual",
                        help='How to compute gradient of group loss, manual or torch')
    
    parser.add_argument('--c', type= float, default= 1,
                        help='objective function relaxation')
    
    parser.add_argument('--s_c', type= float, default= 2,
                        help='scale of objective function relaxation')
    
    # c_s in paper for EWA and PWA
    parser.add_argument('--c_l0', type= float, default= 1, # 100
                        help='LO norm loss relaxation')
    
    # c_s in paper for GWA
    parser.add_argument('--c_grouped', type= float, default= 0,
                        help='grouped loss relaxation')
    
    parser.add_argument('--c_linf', type= float, default= 0, #10
                        help='linf loss relaxation')
    
    parser.add_argument('--p', type= float, default= 1e4, # 1e4
                        help='p in linf loss function, LSEAp')
    
    parser.add_argument('--eta_linf', type= float, default= 1,
                        help='optional for Restricting linf, must be less than 1')
    
    parser.add_argument('--sparse_mode', type=str, default='RGB',help='RGB: number of \
                        pertabeted pixels is important, Gray: l0 is important')
    
    parser.add_argument('--dmax', type= float, default= 1,
                        help='maximum value of perturbation for OSl0')  
    
    parser.add_argument('--s_sikma', type= float, default= 0.5,
                        help='skale of sikma in OSl0')
    
    parser.add_argument('--outer_iter', type= float, default= 10,
                        help='number of steps for algorithm')
    
    parser.add_argument('--inner_iter', type= float, default= 200,
                        help='number of iteration in each step')
    
    parser.add_argument('--learning_rate', type= float, default= 0.001,
                        help='total learning rate for gradiant, mu in the paper')
    
    return parser
