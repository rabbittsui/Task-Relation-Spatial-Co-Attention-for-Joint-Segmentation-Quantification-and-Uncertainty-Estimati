# This code is based on: https://github.com/SimonKohl/probabilistic_unet
import os
import torch
import torch.nn as nn
import numpy as np
from nest_dtask3 import NestedUNet#,nestub2nup#unetnestbn2
from utils1 import init_weights, init_weights_orthogonal_normal, l2_regularisation
import torch.nn.functional as F
from lovasz_loss import LovaszSoftmax
from torchvision.utils import save_image
from functools import partial
from flow_loss import sim_dis_compute
from torchvision.utils import save_image

nonlinearity = partial(F.relu, inplace=True)
#from seglstm import ConvLSTM_

save_path='my_save\\'
if not os.path.exists(save_path):
        os.makedirs(save_path)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=1, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3)
        return scale

def get_kd_cost(source_logits, source_gt, target_logits, target_gt):
    num_classes=source_logits.shape[1]
    kd_loss = 0.0
    source_logits = source_logits
    target_logits = target_logits
    source_prob = []
    target_prob = []
    temperature = 4.0
    for i in range(num_classes):
        eps = 1e-6
        gt_one = source_gt[:, i, :, :]
        s_mask = gt_one.unsqueeze(1)
        s_mask = s_mask.repeat(1, num_classes, 1, 1)
        s_logits_mask_out = source_logits * s_mask
        s_logits_avg = torch.sum(s_logits_mask_out, [0, 2, 3]) / (torch.sum(source_gt[:, i, :, :]) + eps)
        s_soft_prob = F.softmax(s_logits_avg / temperature)
        source_prob.append(s_soft_prob)
        t_one = target_gt[:, i, :, :]
        t_mask = t_one.unsqueeze(1)
        t_mask = t_mask.repeat(1, num_classes, 1, 1)
        t_logits_mask_out = target_logits * t_mask
        t_logits_avg = torch.sum(t_logits_mask_out, [0, 2, 3]) / (torch.sum(target_gt[:, i, :, :]) + eps)
        t_soft_prob = F.softmax(t_logits_avg / temperature)
        target_prob.append(t_soft_prob)

        ## KL divergence loss
        loss = (torch.sum(s_soft_prob * torch.log(s_soft_prob / t_soft_prob)) + torch.sum(
            t_soft_prob * torch.log(t_soft_prob / s_soft_prob))) / 2.0

        ## L2 Norm
        # loss = tf.nn.l2_loss(s_soft_prob - t_soft_prob) / n_class

        kd_loss += loss

    kd_loss = kd_loss / num_classes
    return kd_loss


def _get_segmentation_cost(input, target):
    """
    calculate the loss for segmentation prediction
    :param seg_logits: activations before the Softmax function
    :param seg_gt: ground truth segmentaiton mask
    :return: segmentation loss, according to the cost_kwargs setting, cross-entropy weighted loss and dice loss
    """
    num_classes=input.shape[0]
    seg_logits = input
    seg_gt = target
    softmaxpred = F.softmax(seg_logits, dim=1)

    # calculate dice loss, - 2*interesction/union, with relaxed for gradients back-propagation
    dice = 0
    for i in range(num_classes):
        inse = torch.sum(softmaxpred[:, i, :, :] * seg_gt[:, i, :, :])
        l = torch.sum(softmaxpred[:, i, :, :] * softmaxpred[:, i, :, :])
        r = torch.sum(seg_gt[:, i, :, :])
        dice += 2.0 * inse / (l + r + 1e-7)  # here 1e-7 is relaxation eps
    dice_loss = 1 - dice / num_classes

    # calculate cross-entropy weighted loss
    ce_weighted = 0
    for i in range(num_classes):
        gti = seg_gt[:, i, :, :]
        predi = softmaxpred[:, i, :, :]
        weighted = 1 - (torch.sum(gti) / torch.sum(seg_gt))
        ce_weighted += -1.0 * weighted * gti * torch.log(torch.clamp(predi, 0.005, 1))
    ce_weighted_loss = torch.mean(ce_weighted)

    return (dice_loss + ce_weighted_loss)


def elbo_loss(real_seg1, real_seg2, real_seg3, real_seg4, c1, c2, c3, c4,ed_seg , es_seg , ed_seg2 , es_seg2 ,ed_c, es_c , ed_c2 , es_c2, Cal_out1,Cal_out2,Cal_out3,edv, esv, ef):
    """
    Calculate the evidence lower bound of the log-likelihood of P(Y|X)
    """

    criterion = nn.CrossEntropyLoss(size_average=False, reduce=False, reduction=None)
    kd = get_kd_cost(ed_seg, real_seg1, ed_seg2, real_seg3) + get_kd_cost(es_seg, real_seg2, es_seg2, real_seg4)

    # Here we use the posterior sample sampled above
    seg1_c, seg2_c, seg3_c, seg4_c = real_seg1 * c1, real_seg2 * c2, real_seg3 * c3, real_seg4 * c4
    # kd=get_kd_cost(ed_c, seg1_c, ed_c2, seg3_c)+get_kd_cost(es_c ,seg2_c, es_c2, seg4_c)
    real_seg1, real_seg2, real_seg3, real_seg4 = torch.argmax(real_seg1, dim=1), torch.argmax(real_seg2,
                                                                                              dim=1), torch.argmax(
        real_seg3, dim=1), torch.argmax(real_seg4, dim=1)
    real_seg1c, real_seg2c, real_seg3c, real_seg4c = torch.argmax(seg1_c, dim=1), torch.argmax(seg2_c,
                                                                                               dim=1), torch.argmax(
        seg3_c, dim=1), torch.argmax(seg4_c, dim=1)
    preds1_c, preds2_c, preds3_c, preds4_c = ed_c * c1, es_c * c2, ed_c2 * c3, es_c2 * c4
    preds1_c1, preds2_c1, preds3_c1, preds4_c1 = ed_seg * c1, es_seg * c2, ed_seg2 * c3, es_seg2 * c4

    rankl_loss = criterion(input=preds1_c, target=real_seg1c) + criterion(input=preds2_c,
                                                                          target=real_seg2c) + criterion(input=preds3_c,
                                                                                                         target=real_seg3c) + criterion(
        input=preds4_c, target=real_seg4c)

    reconstruction_loss = 2*criterion(input=ed_seg, target=real_seg1) + 2*criterion(input=es_seg, target=real_seg2) \
                          + criterion(input=ed_seg2, target=real_seg3) + criterion(input=es_seg2, target=real_seg4)

    sim_loss = sim_dis_compute(preds1_c, preds1_c1) + sim_dis_compute(preds2_c, preds2_c1) + sim_dis_compute(preds3_c,
                                                                                                             preds3_c1) + sim_dis_compute(
        preds4_c, preds4_c1)

    reconstruction_loss = torch.sum(reconstruction_loss) + torch.sum(rankl_loss)
    mean_reconstruction_loss = torch.mean(reconstruction_loss)
    criterion1 = nn.MSELoss()
    index_loss = F.smooth_l1_loss(torch.cat([Cal_out1,Cal_out2,Cal_out3],dim=1),torch.cat([edv.float(), esv.float(),ef.float()],dim=1))##criterion1(torch.cat([Cal_out1,Cal_out2,Cal_out3],dim=1),torch.cat([edv.float(), esv.float(),ef.float()],dim=1))
    #index_loss = criterion2(Cal_out1.squeeze(), edv.long().squeeze()) + criterion2(Cal_out2.squeeze(), esv.long().squeeze()) + criterion2(Cal_out3.squeeze(), ef.long().squeeze())
    index_loss1 =F.smooth_l1_loss((Cal_out1- Cal_out2 )/Cal_out1, ef) #criterion1( (Cal_out1- Cal_out2 )/Cal_out1, ef)
    index_loss2 =F.smooth_l1_loss (Cal_out1 - Cal_out2, edv-esv)#criterion1(Cal_out1 - Cal_out2, edv-esv)
    '''
    if iter<20:
        return (mean_reconstruction_loss + kd + 10 * sim_loss)+(index_loss+index_loss1+index_loss2)*1000000#2*(mean_reconstruction_loss + kd + 10 * sim_loss)+index_loss+index_loss1+index_loss2
    else:
        return (mean_reconstruction_loss + kd + 10 * sim_loss)+(index_loss+index_loss1+index_loss2)*10000000#2*(mean_reconstruction_loss + kd + 10 * sim_loss)+index_loss+index_loss1+index_loss2
    '''
    return (mean_reconstruction_loss + kd + 10 * sim_loss) + (
                index_loss + index_loss1 + index_loss2) * 1000000  # 2*(mean_reconstruction_loss + kd + 10 * sim_loss)+index_loss+index_loss1+index_loss2


class Cos_Attn_no(nn.Module):
    def __init__(self, inplanes, num_class):
        super(Cos_Attn_no, self).__init__()

        self.sub_sample = False

        self.in_channels = inplanes
        self.inter_channels = None
        self.num_class = num_class

        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 4
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        bn = nn.BatchNorm2d

        self.g1 = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1)
        self.g2 = conv_nd(in_channels=32, out_channels=self.inter_channels,
                         kernel_size=1)

        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.inter_channels,
                    kernel_size=1),
            bn(self.inter_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.Q = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.inter_channels,
                    kernel_size=1),
            bn(self.inter_channels)
        )
        nn.init.constant_(self.Q[1].weight, 0)
        nn.init.constant_(self.Q[1].bias, 0)

        self.theta = conv_nd(in_channels=32, out_channels=self.inter_channels,
                             kernel_size=1)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1)

        self.concat_project = nn.Sequential(
            nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
            nn.ReLU()
        )

        self.ChannelGate = ChannelGate(self.inter_channels)
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)

    def forward(self, detect, aim):

        batch_size, _, height_a, width_a = aim.shape
        batch_size, _, height_d, width_d = detect.shape

        #####################################find aim image similar object ####################################################
        detect_t=self.g1(detect)
        d_x = detect_t.view(batch_size, self.inter_channels, -1)
        d_x = d_x.permute(0, 2, 1).contiguous()
        aim_t=self.g2(aim)
        a_x = self.g2(aim).view(batch_size, self.inter_channels, -1)
        a_x = a_x.permute(0, 2, 1).contiguous()

        theta_x = self.theta(aim).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(detect).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        x = f
        N = f.size(-1)
        f_div_C = f / N

        f = f.permute(0, 2, 1).contiguous()
        N = f.size(-1)
        fi_div_C = f / N

        non_aim = torch.matmul(f_div_C, d_x)
        non_aim = non_aim.permute(0, 2, 1).contiguous()
        non_aim = non_aim.view(batch_size, self.inter_channels, height_a, width_a)
        non_aim = self.W(non_aim)
        non_aim = non_aim + aim_t

        non_det = torch.matmul(fi_div_C, a_x)
        non_det = non_det.permute(0, 2, 1).contiguous()
        non_det = non_det.view(batch_size, self.inter_channels, height_d, width_d)
        non_det = self.Q(non_det)
        non_det = non_det + detect_t


        ##################################### Response in chaneel weight ####################################################

        c_weight = self.ChannelGate(non_aim)
        act_aim = non_aim * c_weight
        act_det = non_det * c_weight

        return act_det,act_aim#non_det, act_det, act_aim, c_weight


class Seg_R(nn.Module):
    def __init__(self, input_channels,num_class):
        super(Seg_R, self).__init__()
        self.conv1 =  nn.Conv2d(input_channels//2, input_channels//4, kernel_size=1)
        self.conv2 =  nn.Conv2d(input_channels//2, input_channels//4, kernel_size=1)
        self.attn1 = Cos_Attn_no(input_channels, num_class)
        self.attn2 = Cos_Attn_no(input_channels, num_class)
        self.up_f = nn.UpsamplingNearest2d(size=[14,14])#[7,7]


    def forward(self,x2_s,x4_s, x2,x4):
        index_c ,index_s= self.attn1(x2,x2_s)
        index_c2 ,index_s2= self.attn2(x4, x4_s)
        #######      two views attention    #############
        x2f = torch.cat([ self.up_f(index_c), self.up_f(index_s)], dim=1)
        x4f = torch.cat([ self.up_f(index_c2) ,self.up_f(index_s2)], dim=1)
        x2f_v1= self.conv1(x2f)
        x4f_v1= self.conv2(x4f)
        return x2f_v1,x4f_v1,index_s,index_s2

class Index_Cal1(nn.Module):
    def __init__(self, input_channels):
        super(Index_Cal1, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv_r1 =  nn.Conv2d(input_channels*2, input_channels, kernel_size=1)
        self.conv_r2 =  nn.Conv2d(input_channels*2, input_channels, kernel_size=1)
        self.conv_r3 = nn.Conv2d(input_channels * 2, input_channels, kernel_size=1)
        self.conv_r4 = nn.Conv2d(input_channels * 2, input_channels, kernel_size=1)
        self.conv_r5 = nn.Conv2d(input_channels * 2, input_channels, kernel_size=1)
        self.Flatten_view=Flatten()

        self.fc1 = nn.Sequential(nn.Linear(input_channels*14 * 14, input_channels), nn.ReLU(True),nn.Linear(input_channels, 64),nn.ReLU(True),nn.Linear(64, 1))#input_channels*7*7

        #self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)


    def forward(self, a2c,a4c):
        Cal_in = torch.cat([a2c,a4c],dim=1)
        r1=F.softmax(self.relu(self.conv_r1(Cal_in)))
        r2=F.softmax(self.relu(self.conv_r2(Cal_in)))
        a2c_in= a2c*r1
        a4c_in= a4c*r2
        x2f = torch.cat([a2c, a4c_in], dim=1)
        x4f = torch.cat([a4c, a2c_in], dim=1)
        x2f4 = self.relu(self.conv_r3(x2f))
        x4fc2 = self.relu(self.conv_r4(x4f))
        x_CalIn = torch.cat([x2f4,x4fc2],dim=1)
        x_flatten=(self.relu(self.conv_r5(x_CalIn)))
        x_view = self.Flatten_view(x_flatten)
        Cal_out1 =F.sigmoid(self.fc1(x_view))
        return x2f4,x4fc2,Cal_out1



class MyNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=3, beta=0.5):
        super(MyNet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.beta = beta
        self.padding = True,
        self.net = NestedUNet(self.num_classes, self.input_channels)
        self.inter_channels=32
        self.final = nn.Conv2d(self.inter_channels, self.num_classes, kernel_size=1)
        self.final_flow = nn.Conv2d(self.inter_channels, self.num_classes, kernel_size=1)
        self.Flatten = Flatten()
        self.R_l1=Seg_R(512,self.num_classes)
        self.R_l2=Seg_R(512,self.num_classes)
        self.Cal_Out1=Index_Cal1(input_channels=128)
        self.Cal_Out2=Index_Cal1(input_channels=128)
        self.relu = nn.ReLU(inplace=True)
        self.in_channels = 128
        self.conv_r6 =nn.Conv2d(self.in_channels * 4, self.in_channels, kernel_size=1)
        self.conv_t1 = nn.Conv2d(self.in_channels,self.inter_channels, kernel_size=1)
        self.conv_t2 = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)
        self.fc2 = nn.Sequential(nn.Linear(self.in_channels * 14 * 14, self.in_channels), nn.ReLU(True),
                                nn.Linear(self.in_channels, 64), nn.ReLU(True), nn.Linear(64, 1))
    def forward(self, patch1,patch2, patch3,patch4):
        self.ed_f,self.ed_f2 ,self.ed_m,self.ed_m2=self.net.forward(patch1,patch3)#self.ed_seg,self.ed_seg2,self.ed_c,self.ed_c2 ,
        self.es_f,self.es_f2,self.es_m,self.es_m2 =self.net.forward(patch2,patch4)#self.es_seg,self.es_seg2,self.es_c,self.es_c2 ,
        x = self.ed_f
        '''
        for l in range(x.shape[1]):
            save_image(x[0, l, :, :].data.cpu(), save_path + "/" + str(l) + "edm" + 'cfeature.png')
        y = self.ed_m
        for kk in range(y.shape[1]):
            save_image(y[0, kk, :, :].data.cpu(), save_path + "/" + str(kk) + "edf_" + 'cfeature.png')
        '''
        self.Cal_In1,self.Cal_In11,self.ed_2f,self.ed_4f=self.R_l1(self.ed_m,self.ed_m2 ,self.ed_f,self.ed_f2)
        self.Cal_In11, self.Cal_In22,self.es_2f,self.es_4f = self.R_l2(self.es_m,self.es_m2,self.es_f,self.es_f2)
        self.ed_seg, self.ed_seg2 = self.final(self.conv_t1(self.ed_2f)),self.final(self.conv_t2(self.ed_4f))
        self.es_seg, self.es_seg2 = self.final(self.conv_t1(self.es_2f)),self.final(self.conv_t2(self.es_4f))
        self.ed_c, self.ed_c2 = self.final_flow(nonlinearity(self.conv_t1(self.ed_2f))),self.final_flow(nonlinearity(self.conv_t2(self.ed_4f)))
        self.es_c, self.es_c2 = self.final_flow(nonlinearity(self.conv_t1(self.es_2f))),self.final_flow(nonlinearity(self.conv_t2(self.es_4f)))
        self.x2f4, self.x4fc2, self.Cal_out10 = self.Cal_Out1( self.Cal_In1,self.Cal_In11)
        self.x2f41, self.x4fc21, self.Cal_out20 = self.Cal_Out2(self.Cal_In11, self.Cal_In22)
        EF_input = torch.cat([self.x2f4, self.x4fc2,self.x2f41, self.x4fc21],dim=1)
        ef_in=self.conv_r6(EF_input)
        self.Cal_out3 =  self.fc2(self.Flatten(self.relu(ef_in)))
        return self.ed_seg, self.es_seg,self.ed_seg2, self.es_seg2,self.ed_c,self.es_c,self.ed_c2,self.es_c2,self.Cal_out10,self.Cal_out20,self.Cal_out3

if __name__ == "__main__":

    model = Index_Cal1(input_channels=1, num_classes=4)
    model.eval()
    input = torch.rand(1, 2, 112, 112)
    output = model(input)
    print(output.shape())