#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import nibabel as nib
import glob
import os
import torchvision.transforms as transforms
from skimage import transform
import matplotlib.pyplot as plt



def normalise_image(image):
    '''
    make image zero mean and unit standard deviation
    '''

    img_o = np.float32(image.copy())
    m = np.mean(img_o)
    s = np.std(img_o)
    return np.divide((img_o - m), s)



def crop_or_pad_slice_to_size(slice, nx=224, ny=224):

    x, y = slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_cropped



def data_prepare (nx=224, norm=True):
    num=0
    imageS=[]
    img_T=[]
    mask_T=[]
    imageSS=[]
    filename3D=[]
    imgC=glob.glob(os.path.join('datacollect', 'patient???_frame??.nii.gz'))
    mskC=glob.glob(os.path.join('datacollect', 'patient???_frame??_gt.nii.gz'))
    imgC.sort()
    mskC.sort()
    stack=0
    for i in range(200):
        image_name=imgC[i]
        mask_name=mskC[i]
        image = nib.load(image_name).get_data()
        mask = nib.load(mask_name).get_data()
    
        imageS.append(image.shape[-1])
        num=num+imageS[i]    
    
        target_resolution = (1.37, 1.37)
        
        
        img = image
        img = normalise_image(img)
    
        n1_header = nib.load(image_name).header
        pixel_size=n1_header['pixdim'][1:3] 


        
        scale_vector = [pixel_size[0] / target_resolution[0], pixel_size[1] / target_resolution[1]]

        for zz in range(img.shape[2]):

            slice_img = np.squeeze(img[:, :, zz])
            slice_rescaled = transform.rescale(slice_img,scale_vector, order=1, 
                                               preserve_range=True,multichannel=False,mode = 'constant')

            slice_mask = np.squeeze(mask[:, :, zz])
            mask_rescaled = transform.rescale(slice_mask,scale_vector,order=0,
                                              preserve_range=True,multichannel=False,mode='constant')

            slice_cropped = crop_or_pad_slice_to_size(slice_rescaled, nx, nx)
            mask_cropped = crop_or_pad_slice_to_size(mask_rescaled, nx, nx)

            img_T.append(slice_cropped)
            mask_T.append(mask_cropped)
            if zz!=0 and zz!=img.shape[2]-1:
                imageSS.append(stack)
                filename3D.append(image_name)
       
            stack=stack+1
    

    img_T=np.float32(np.array(img_T));    
    mask_T=np.float32(np.array(mask_T));
    if norm==True:
        mask_T=mask_T/np.max(mask_T)  ################undo
    return img_T, mask_T, imageSS, num
    
def dice_coefficient(predicted, target):

    smooth = 1e-5
    product = np.multiply(predicted, target)
    intersection = np.sum(product)
    coefficient = (2 * intersection + smooth) / (np.sum(predicted) + np.sum(target) + smooth)
    return coefficient


def history_plot(history):
    plt.figure(figsize=(20, 10))
    plt.title('Loss Over Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    train_curve = plt.plot(history['train_loss'], marker = 'o', label = 'Train loss')
    validation_curve = plt.plot(history['validation_loss'], marker = 'o', label = 'Validation loss')
    plt.legend(fontsize = 15)
    plt.show()



    
def plot_result(image, mask, output, title, transparency = 0.38, save_path = None):


    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(
        20, 15), gridspec_kw={'wspace': 0.025, 'hspace': 0.010})
    fig.suptitle(title, x=0.5, y=0.92, fontsize=20)
    
    plt.subplot(2, 3, 1)
    plt.title("Original Mask", fontdict={'fontsize': 16})
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.title("Predicted Mask", fontdict={'fontsize': 16})
    plt.imshow(output, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    mask_diff = np.abs(np.subtract(mask, output))
    plt.title("Mask Difference", fontdict={'fontsize': 16})
    plt.imshow(mask_diff, cmap='gray')
    plt.axis('off')

    seg_output = mask*transparency
    seg_image = np.add(image, seg_output)/2
    plt.subplot(2, 3, 4)
    mask_diff = np.abs(np.subtract(mask, output))
    plt.title("Original Segmentation", fontdict={'fontsize': 16})
    plt.imshow(seg_image, cmap='gray')
    plt.axis('off')
    
    seg_output = output*transparency
    seg_image = np.add(image, seg_output)/2
    plt.subplot(2, 3, 5)
    mask_diff = np.abs(np.subtract(mask, output))
    plt.title("Predicted Segmentation", fontdict={'fontsize': 16})
    plt.imshow(seg_image, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    mask_diff = np.abs(np.subtract(mask, output))
    plt.title("Original Input Image", fontdict={'fontsize': 16})
    plt.imshow(image, cmap='gray')
    plt.axis('off')


    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi = 90, bbox_inches = 'tight')

    plt.show()    
    
def evulateData():
    num=0
    ED=[]
    ES=[]
    T3=[]
    T1=[]
    imageS=[]
    stack=0
    imgC=glob.glob(os.path.join('datacollect', 'patient???_frame??.nii.gz'))
    mskC=glob.glob(os.path.join('datacollect', 'patient???_frame??_gt.nii.gz'))
    imgC.sort()
    mskC.sort()
    stack=0
    for i in range(95*2, 200):
        image_name=imgC[i]
        mask_name=mskC[i]
        image = nib.load(image_name).get_data()
        mask = nib.load(mask_name).get_data()
        imageS.append(image.shape[-1])
        num=num+imageS[stack]
        if (i%2==0):
            #ED.append(range(1,num))
            ED.extend(list(range(num-image.shape[-1]-stack*2-2, num-4-stack*2)))
            ES.extend(list(range( num-4-stack*2,num-4-stack*2+image.shape[-1]-2)))

        if (stack//2==3 or stack//2==0):
            if (i%2==0):
                T3.extend(list(range(num-image.shape[-1]-stack*2-2, num-4-stack*2+image.shape[-1]-2)))
        else:
            if (i%2==0):
                T1.extend(list(range(num-image.shape[-1]-stack*2-2, num-4-stack*2+image.shape[-1]-2)))
        stack=stack+1
    T3=T3[2:-1]
    ED=ED[2:-1]   
    return ED,ES,T1,T3    

