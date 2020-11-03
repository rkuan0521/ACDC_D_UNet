#!/usr/bin/env python
# coding: utf-8

# In[ ]:



from torch import nn
from torch.nn import functional as F
from torchvision import models
import torch
import numpy as np

ratio=0.7

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            #nn.ReLU(inplace = True),
            nn.BatchNorm2d(ch_out, track_running_stats=True),
            nn.Conv2d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            #nn.ReLU(inplace = True),
            nn.BatchNorm2d(ch_out, track_running_stats=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class conv_block3D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            #nn.ReLU(inplace = True),
            nn.BatchNorm3d(ch_out, track_running_stats=True),
            nn.Conv3d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            #nn.ReLU(inplace = True),
            nn.BatchNorm3d(ch_out, track_running_stats=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x    

    
class Dimension_1x1x1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Dimension_1x1x1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size = 1, stride = 1, padding = 0))

    def forward(self,x):
        x = self.conv(x)
        return x 
    
    
class Dimension_Transform_Block_pre(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Dimension_Transform_Block_pre, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.ReLU(inplace = True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x     
    
class Dimension_Transform_Block(nn.Module):
    def __init__(self, ch_inout, ch_mid, kernel_size):
        super(Dimension_Transform_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.AvgPool2d(kernel_size),
            nn.Flatten(),
            nn.Linear(ch_inout, ch_mid),
            nn.ReLU(inplace = True),
            nn.Linear(ch_mid, ch_inout),
            #nn.ReLU(inplace = True)
            nn.Sigmoid()
            #nn.ReLU(inplace = True) #nn.Sigmoid()
        )

    def forward(self,x):
        x = self.conv(x)
        return x     
    
    
    
    
class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.ReLU(inplace = True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
    
    
    
class U_Net(nn.Module):


    def __init__(self, img_ch = 3, output_ch = 1, nx =224, first_layer_numKernel = 32):
        '''
        Constructor for UNet class.
        Parameters:
            img_ch(int): Input channels for the network. Default: 1
            output_ch(int): Output channels for the final network. Default: 1
            first_layer_numKernel(int): Number of kernels uses in the first layer of our unet.
        '''
        super(U_Net, self).__init__()
        #super().__init__()
        self.nx=nx
        self.Maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.Maxpool3D = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.Maxpool3D2 = nn.MaxPool3d(kernel_size = [1, 2, 2], stride = 2)
        self.GlobalPool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.Conv1 = conv_block(ch_in = img_ch, ch_out = first_layer_numKernel)
        self.Conv1D = conv_block3D(ch_in = 1, ch_out = first_layer_numKernel)
        self.Conv2 = conv_block(ch_in = first_layer_numKernel, ch_out = 2 * first_layer_numKernel)
        self.Conv2D = conv_block3D(ch_in = first_layer_numKernel, ch_out = 2 * first_layer_numKernel)
        self.Conv3 = conv_block(ch_in = 2 * first_layer_numKernel, ch_out = 4 * first_layer_numKernel)
        self.Conv3D = conv_block3D(ch_in = 2 * first_layer_numKernel, ch_out = 4 * first_layer_numKernel)
        self.Conv4 = conv_block(ch_in = 4 * first_layer_numKernel, ch_out = 8 * first_layer_numKernel)
       #self.Conv4D = conv_block3D(ch_in = 8 * first_layer_numKernel, ch_out = 16 * first_layer_numKernel)
        self.Conv5 = conv_block(ch_in = 8 * first_layer_numKernel, ch_out = 16 * first_layer_numKernel)

        self.Up5 = up_conv(ch_in = 16 * first_layer_numKernel, ch_out = 8 * first_layer_numKernel)
        self.Up_conv5 = conv_block(ch_in = 16 * first_layer_numKernel, ch_out = 8 * first_layer_numKernel)

        self.Up4 = up_conv(ch_in = 8 * first_layer_numKernel, ch_out = 4 * first_layer_numKernel)
        self.Up_conv4 = conv_block(ch_in = 8 * first_layer_numKernel, ch_out = 4 * first_layer_numKernel)
        
        self.Up3 = up_conv(ch_in = 4 * first_layer_numKernel, ch_out = 2 * first_layer_numKernel)
        self.Up_conv3 = conv_block(ch_in = 4 * first_layer_numKernel, ch_out = 2 * first_layer_numKernel)
        
        self.Up2 = up_conv(ch_in = 2 * first_layer_numKernel, ch_out = first_layer_numKernel)
        self.Up_conv2 = conv_block(ch_in = 2 * first_layer_numKernel, ch_out = first_layer_numKernel)

        self.Conv_1x1 = nn.Conv2d(first_layer_numKernel, 4, kernel_size = 1, stride = 1, padding = 0)
        
        self.deform1=Dimension_1x1x1(ch_in = 64, ch_out = 1)
        
        self.prep1 = Dimension_Transform_Block_pre(ch_in = 1, ch_out = 64)

        #int(np.ceil(64/ratio))
        #ch_inout, ch_mid, kernel_size
        self.Transform1=Dimension_Transform_Block(ch_inout = 64, ch_mid = 32, kernel_size = int(np.ceil(nx/2)))
 
        self.deform2=Dimension_1x1x1(ch_in = 128, ch_out = 1)
        self.prep2 = Dimension_Transform_Block_pre(ch_in = 1, ch_out = 128)
        # int(np.ceil(128/ratio))
        self.Transform2=Dimension_Transform_Block(ch_inout = 128, ch_mid = 64, kernel_size = int(np.ceil(nx/4)))
        self.dropout=nn.Dropout(p=0.1)


    def forward(self, x, xx):

        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1) # 2 32 58 58
        xx1 = self.Conv1D(xx) 
        xx2 = self.Maxpool3D(xx1)  #2 32 14 58 58      2, 32, 3, 116, 116  
        x2 = self.Conv2(x2)
        
        
        ########################################################
        ########################################################
        xx2a = self.Conv2D(xx2) #2 64 14 58 58    2, 64, 1, 58, 58
        xx2b = self.deform1(xx2a)  # 2, 1, 14, 58, 58   2, 1, 1, 58, 58
        xx2c = torch.squeeze(xx2b,1)  # 2, 14, 58, 58  2, 1, 58, 58
        xx2d = self.prep1(xx2c)  # 2, 64, 58, 58   2, 64, 58, 58
        xx2e = self.Transform1(xx2d) # 2, 64
        x2Copy = x2
        x2r = self.Transform1(x2Copy) #2, 64 


        
        x2rr=torch.stack([x2r],dim=2)
        x2rr=torch.stack([x2rr],dim=2)  # 2, 64, 58, 58  112    
        x2rr=torch.cat([x2rr]*112, 3)
        x2rr=torch.cat([x2rr]*112, 2)

        xx2ee=torch.stack([xx2e],dim=2)
        xx2ee=torch.stack([xx2ee],dim=2) #2, 64, 58, 58
        xx2ee=torch.cat([xx2ee]*112, 3)
        xx2ee=torch.cat([xx2ee]*112, 2)    

        x2=torch.mul(x2rr,x2)+torch.mul(xx2ee,xx2d)  # 2, 64, 58, 58
        x3 = self.Maxpool(x2)   
        
       # xx2 2, 32, 1, 58, 58     
        xx3 = self.Maxpool3D2(xx2a) # 2, 32, 7, 29, 29    2, 32, 1, 29, 29 
        #xx3 = self.Conv3D(xx3) # 2, 64, 7, 29, 29        2, 64, 1, 29, 29
        

        ########################################################
        ########################################################
        #logging.info("Deform2") 
        xx3a = self.Conv3D(xx3) #2, 128, 7, 29, 29   2, 128, 1, 29, 29   5, 128, 1, 56, 56  
        xx3b = self.deform2(xx3a) #2, 1, 7, 29, 29     2, 1, 1, 29, 29    5, 1, 1, 56, 56
        xx3c = torch.squeeze(xx3b, 1) # 2, 7, 29, 29        5, 1, 56, 56
        xx3d = self.prep2(xx3c) # 2, 128, 29, 29      5, 128, 56, 56
        xx3e = self.Transform2(xx3d) # 2, 128   5, 128
        
        x3 = self.Conv3(x3) # 2, 128, 29, 29      
        
        x3Copy = x3
        x3r = self.Transform2(x3Copy)        
        x3rr=torch.stack([x3r],dim=2)
        x3rr=torch.stack([x3rr],dim=2)
        x3rr=torch.cat([x3rr]*56, 3)
        x3rr=torch.cat([x3rr]*56, 2)
        
        
        xx3ee=torch.stack([xx3e],dim=2)
        xx3ee=torch.stack([xx3ee],dim=2)

        xx3ee=torch.cat([xx3ee]*56, 3)
        xx3ee=torch.cat([xx3ee]*56, 2)              


        x3=torch.mul(x3rr,x3)+torch.mul(xx3ee,xx3d)
        x4 = self.Maxpool(x3)  # 5, 128, 28, 28
        x4 = self.Conv4(x4)   
        x4 = self.dropout(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        x5 = self.dropout(x5)
        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim = 1)
        
        d5 = self.Up_conv5(d5)
        d4 = self.Up4(d5)

        x33=x3 #[:,:,0:28,0:28]
        d4 = torch.cat((x33, d4), dim = 1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x22=x2#[:,:,0:56,0:56]
        d3 = torch.cat((x22, d3), dim = 1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x11=x1#[:,:,0:112,0:112]
        d2 = torch.cat((x11, d2), dim = 1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        #output=torch.zeros(5, 1, self.nx, self.nx)
        output = F.sigmoid(d1)
        #output = F.softmax(d1, dim=1)
        #print(output.shape)
        return output

