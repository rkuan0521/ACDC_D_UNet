#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
from torch.utils import data as D

class data2D3D(D.Dataset):
   
    def __init__(self, root, imageSS, img_T, mask_T, num, nx, aug = False):
        self.root = root
        self.imageSS = imageSS
        self.img_T = img_T
        self.mask_T = mask_T
        self.num = num
        self.nx = nx
        
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        imgTemp = np.zeros((  self.nx, self.nx, 3), dtype=np.float32)
        maskTemp = np.zeros(( self.nx, self.nx, 3), dtype=np.float32)

        slice_vol = np.zeros(( 3, self.nx, self.nx), dtype=np.float32)
        mask_vol = np.zeros(( 1, self.nx, self.nx), dtype=np.float32)
        it=self.imageSS[index]

        
        slice_vol[0,:,:]=np.float32(self.img_T[it-1,:,:])
        slice_vol[1,:,:]=np.float32(self.img_T[it+1,:,:])
        slice_vol[2,:,:]=np.float32(self.img_T[it,:,:])
        mask_vol=np.float32(self.mask_T[it,:,:])
        
        image2D=torch.from_numpy(slice_vol).type(torch.FloatTensor).view( 3, self.nx, self.nx)
        mask2D=torch.from_numpy(mask_vol).type(torch.FloatTensor).view( 1, self.nx, self.nx)

        image3D=image2D.view( 1, 3, self.nx, self.nx)
     
        sample = {'index': int(index), 'image2D': image2D, 'mask2D': mask2D, 'image3D': image3D}
            
        return sample
   
    def __len__(self):

        return self.num-200*2

