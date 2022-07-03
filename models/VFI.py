

import torch
from math import pi
import torch.nn as nn
import torch.nn.functional as F
from .flow_net.flow_Unet import UNet
from .flow_net.FastFlow_Net import FastFlowNet, backwarp
from .softsplat import ModuleSoftsplat as ForwardWarp
from .modules.GridNet import GridNet
# from .modules.update import UpdateBlock
# from .modules.context import ContextNet


class FeatureExtractor(nn.Module):
    """The quadratic model"""
    def __init__(self, path='./network-default.pytorch'):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.prelu3 = nn.PReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.prelu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(64, 96, 3, stride=2, padding=1)
        self.prelu5 = nn.PReLU()
        self.conv6 = nn.Conv2d(96, 96, 3, padding=1)
        self.prelu6 = nn.PReLU()

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x1 = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x1))
        x2 = self.prelu4(self.conv4(x))
        x = self.prelu5(self.conv5(x2))
        x3 = self.prelu6(self.conv6(x))

        return x1, x2, x3



class VFINet(nn.Module):
    def __init__(self, args):
        super(VFINet, self).__init__()
        self.t = 0.5

        self.attention = args.attention

        # if args.mode == 'train':
        #     self.training = True
        # else:
        #     self.training = False

        # self.flow_estimator = UNet(6, 6)
        self.flow_estimator = FastFlowNet()
        self.feature_extactor = FeatureExtractor()
        self.forward_warp = ForwardWarp('average')
        self.synnet = GridNet(6, 64, 128, 96*2, 3, self.attention)


    def flow_scale(self, flo, shape):
        flow = F.interpolate(flo, shape, mode='bilinear')
        flow[:, 0, :, :] = flow[:, 0, :, :].clone() * shape[1] / flo.size()[3]
        flow[:, 1, :, :] = flow[:, 1, :, :].clone() * shape[0] / flo.size()[2]

        return flow

    def cal_inter_flow(self, flo, sigma, t):
        alpha = torch.atan2(flo[:,1:,:,:], flo[:,0:1,:,:])
        beta = torch.asin(sigma)
        d = torch.sqrt(flo[:,1:,:,:]**2 + flo[:,0:1,:,:]**2)
        
        mask = torch.abs(sigma) < 0.01
        R = torch.zeros(d.shape).float().cuda()
        R[~mask] = d[~mask] / (2*sigma[~mask])
        theta_0 = alpha + pi/2 + beta
        theta_1 = alpha + pi/2 - beta
        theta_t = theta_0 + (theta_1 - theta_0) * t
        flow_tx = R * (torch.cos(theta_t) - torch.cos(theta_0))
        flow_tx[mask] = t * flo[:,0:1,:,:][mask]
        flow_ty = R * (torch.sin(theta_t) - torch.sin(theta_0)) 
        flow_ty[mask] = t * flo[:,1:,:,:][mask]
        flow_t = torch.cat([flow_tx, flow_ty], dim=1)

        # flow_t = torch.where(torch.isnan(flow_t), t*flo, flow_t)
        # flow_t = torch.where(torch.isinf(flow_t), t*flo, flow_t)

        return flow_t 

    def forward(self, img0, img1, mode='train'):
        
        B, _, H, W = img0.shape

        mean = torch.cat([img0,img1],dim=1).mean(dim=[1,2,3], keepdim=True)
        img0 = img0 - mean.expand(img0.size())
        img1 = img1 - mean.expand(img1.size())
        t = self.t

        # flow, sigma = self.flow_estimator(torch.cat([img0,img1],dim=1))
        # flow_01, flow_10 = torch.split(flow, [2,2], dim=1)
        # sigma_01, sigma_10 = torch.split(0.667*sigma, [1,1], dim=1)
        
        
        flow_01, sigma_01 = self.flow_estimator(torch.cat([img0,img1],dim=1))
        flow_10, sigma_10 = self.flow_estimator(torch.cat([img1,img0],dim=1))
        flow_0t = self.cal_inter_flow(flow_01, sigma_01, t)
        flow_1t = self.cal_inter_flow(flow_10, sigma_10, 1-t)
  
        # flow_0t = t*flow_01
        # flow_1t = (1-t)*flow_10
        feat01, feat02, feat03 = self.feature_extactor(img0)
        feat11, feat12, feat13 = self.feature_extactor(img1)


        flow_0t_d = self.flow_scale(flow_0t, feat01.shape[2:])
        flow_1t_d = self.flow_scale(flow_1t, feat11.shape[2:])

        flow_0t_dd = self.flow_scale(flow_0t, feat02.shape[2:])
        flow_1t_dd = self.flow_scale(flow_1t, feat12.shape[2:])

        flow_0t_ddd = self.flow_scale(flow_0t, feat03.shape[2:])
        flow_1t_ddd = self.flow_scale(flow_1t, feat13.shape[2:])

        img_0t = self.forward_warp(img0, flow_0t_d)
        feat_0t_1 = self.forward_warp(feat01, flow_0t_d)
        feat_0t_2 = self.forward_warp(feat02, flow_0t_dd)
        feat_0t_3 = self.forward_warp(feat03, flow_0t_ddd)

        img_1t = self.forward_warp(img1, flow_1t_d)
        feat_1t_1 = self.forward_warp(feat11, flow_1t_d)
        feat_1t_2 = self.forward_warp(feat12, flow_1t_dd)
        feat_1t_3 = self.forward_warp(feat13, flow_1t_ddd)

        if self.attention:
            sigma_0t = self.forward_warp(sigma_01, flow_0t)
            sigma_1t = self.forward_warp(sigma_10, flow_1t)  
            sigma = torch.cat([sigma_0t, sigma_1t], dim=1) 
            # sigma = F.interpolate(sigma, img0.shape[2:], mode='bilinear')
            img_t = self.synnet(torch.cat([img_0t, img_1t], dim=1), torch.cat([feat_0t_1, feat_1t_1], dim=1), torch.cat([feat_0t_2, feat_1t_2], dim=1), torch.cat([feat_0t_3, feat_1t_3], dim=1), sigma)
        else:
            img_t = self.synnet(torch.cat([img_0t, img_1t], dim=1), torch.cat([feat_0t_1, feat_1t_1], dim=1), torch.cat([feat_0t_2, feat_1t_2], dim=1), torch.cat([feat_0t_3, feat_1t_3], dim=1))

        img_t += mean.expand(img_t.size())
        if mode == 'train':
            img_01 = backwarp(img0, self.flow_scale(flow_10, img0.shape[2:]))
            img_10 = backwarp(img1, self.flow_scale(flow_01, img1.shape[2:]))
            img0_err = img0 - img_10
            img1_err = img1 - img_01
            return img_t, img0_err, img1_err
        else:
            return img_t