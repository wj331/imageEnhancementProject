import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np

#color constancy loss
#To promote color balance distribution, desirable in image processing tasks to prevent color casts or tints
class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):
        #split into batch, channels, height and width
        b,c,h,w = x.shape

        #calculates mean of x across dimensions height(2) and width(3)
        mean_rgb = torch.mean(x,[2,3],keepdim=True)

        #splits mean_rgb (mean of red, green and blue) along the channel dimension
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)

        #squared differences between each pair of colour
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)

        #L2 norm (euclidean distance) of the vector
        #distance provides a single scalar value representing overall disparity between the three colours
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)

        return k
    
#spatial consistency loss
#To ensure enhanced image maintains spatial structure of the original
class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        #kernels designed to compute gradients in the left,right,up and down directions
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)

        #wraps each kernel in a Parameter, gradient not needed so no change
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)

        #downsample the input by taking average over non-overlapping 4x4 regions
        self.pool = nn.AvgPool2d(4)
    
    def forward(self, org , enhance):
        b,c,h,w = org.shape

        #computes mean of original and enhanced images across the channel dimension (dimension 1)
        #effectively converts to grayscale by averaging RGB channels
        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        #downsampling: reduces spatial dimensions by a factor of 4
        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)

        #weighting factor based on difference between original and threshold value of 0.3
        #calculation emphasizes areas where original pooled intensity is below a threshold(0.3), ensuring enhancements in darker regions are handled carefully
        weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        
        #error calculation: difference between enhanced and original images * sign of difference between enhanced and 0.5
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)

        #Gradient Computation with Directional Kernels
        #original image gradients
        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        #enhanced image gradients
        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)
    
        #gradient difference squared
        #gradient diff penalizes discrepancies in spatial structures, promoting consistency
        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)

        #comprehensive metric of spatial inconsistency between original and enhanced images
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E
    
#exposure control loss    
#measures how well the average brightness(exposure) of an image matches target value(mean_value)
class L_exp(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val #target value we aim to match

    def forward(self, x ):

        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True) #brightness across channels
        #reduces number of channels from 3 to 1 by averaging
        #result is grayscale image of shape (b, 1, h, w)

        #divides image into patches of size patch_size x patch_size
        #computes mean of each patch
        mean = self.pool(x) #tensor of shape (b, 1, h/patch_size, w/patch_size)

        #exposure control loss: penalizes difference between mean and target value
        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2))
        return d

#illumination smoothness loss: used to enforce smoothness in images/feature maps
#ensures smooth transitions between adjacent pixels, reducing noise but preserving edges
class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        #to control importance of smoothness relative to other losses (0.5 = less contribution)
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]

        #count horizontal pairs
        count_h =  (h_x-1) * w_x

        #count vertical pairs
        count_w = h_x * (w_x - 1)

        #horizontal difference between adjacent pixels I(i + 1,j)-I(i,j) (each pixel and its right neighbor)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()

        #vertical difference between adjacent pixels, I(i,j+1)-I(i,j)
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()

        #normalize horizontal TV loss by dividing by total number of horizontal pixel pairs
        #normalize vertical TV loss by dividing by total number of vertical pixel pairs

        #divides by batch size to compute average TV loss across the batch
        return self.TVLoss_weight*2*( h_tv/count_h + w_tv/count_w )/batch_size

#Spatial consistency: neighbouring pixels should have similar properties (color, intensity)
#By minimizing this loss, model encourages smooth transitions in image while preserving important structural details like edges
class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()

    def forward(self, x):
        b,c,h,w = x.shape #note c = 3 for RGB images

        #split into 3 color channels
        r,g,b = torch.split(x , 1, dim=1)

        #mean of each channel across height and width
        mean_rgb = torch.mean(x,[2,3],keepdim=True)

        #split mean_rgb into 3 mean values
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)

        #difference between each color channel and its mean
        #when difference is small, pixel values close to their average
        #reduces abrupt changes between neighbouring pixels
        Dr = r-mr
        Dg = g-mg
        Db = b-mb
        k =torch.pow( torch.pow(Dr,2) + torch.pow(Db,2) + torch.pow(Dg,2),0.5)        
        
        #mean of magnitudes across all pixels in the batch
        k = torch.mean(k)
        return k
    
#difference between 2 images, in terms of high level feature representations
class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        #extract hierarchical features
        #early layers capture low level details like edges
        #deeper layers capture high level features like textures
        features = vgg16(pretrained=True).features #convolutional layers of VGG16

        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4): #layers 0 - 3
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9): #layers 4-8
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False #only want to extract features, not update them

    def forward(self, x): #pass input image x through VGG16 layers
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h

        return h_relu_4_3
