
"""
@author: Alireza Heshmati
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from robustbench.utils import load_model
import timm

class CIFAR_10_net(nn.Module):
    def __init__(self):
        super(CIFAR_10_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        
        x = torch.randn(3,32,32).view(-1,3,32,32)
        self._to_linear = None
        self.convs(x)
        
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256, 10)

        self.dropout = nn.Dropout(p=0.5)

    def convs(self, x):
        # max pooling over 2x2
        x,ind = F.relu(self.conv1(x)), (2, 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x,ind = F.relu(self.conv3(x)), (2, 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.reshape(-1, self._to_linear)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        logit = self.fc3(x)
        return logit
    
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.nirmalizing_input = transforms.Normalize(mean, std)

    def forward(self, input):
        return self.nirmalizing_input(input)
    
def pretrained_model(model_name):
    if  model_name == 'convnet':
        net = CIFAR_10_net()
        net.load_state_dict(
            torch.load('./supplies/cifar_best.pth'))
        normalize_layer= Normalize((0.5, 0.5, 0.5), (1, 1, 1))
        net = nn.Sequential(normalize_layer,net)
        
    elif model_name == 'Liu2023Comprehensive_Swin-L':
        # Load a model from the model zoo
        net = load_model(model_name = model_name,
                    dataset='imagenet',
                    threat_model='Linf')
    
    elif model_name == 'vits16' :
        net = timm.create_model('vit_small_patch16_224', pretrained=True)
        normalize_layer= Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        net = nn.Sequential(normalize_layer,net)

    elif model_name == 'resnet152':
        net = timm.create_model('resnet152.a1_in1k', pretrained=True)
        normalize_layer= Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        net = nn.Sequential(normalize_layer,net)
    else:
        net = load_model(model_name=model_name,
                    dataset='cifar10',
                    threat_model='Linf')
        
    return net
