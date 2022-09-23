from __future__ import print_function, division
import os
import torch
from optparse import OptionParser
import torch
from torch import optim
import sys
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dtask_61 import MyNet,elbo_loss#nestu_dual1
from utils1 import l2_regularisation
from torchvision import transforms
from datagen_gan1 import ListDataset
import torch.nn as nn
import torch.nn.functional as F
import  numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
from torchvision.utils import save_image
import math

def _eval_dice( pred_y,gt_y):
    dice =0
    pred_y=torch.argmax(F.softmax(pred_y,dim=1), 1).cpu()

    for cls in range(1, 4):

        gt =gt_y[:,cls,:,:].cpu().numpy()
        pred = np.zeros(pred_y.shape)

        #gt[gt_y == cls] = 1
        pred[pred_y == cls] = 1

        dice_this = 2*np.sum(gt*pred)/(np.sum(gt)+np.sum(pred))
        dice=dice+dice_this
    return dice/3

def eval_gan_nest(net, dataset, gpu=True):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot1 = 0
    tot2 = 0
    err_edv=0
    err_esv=0
    err_ef=0
    for i, (img1, img2, mask1,mask2, img11, img22, mask11, mask22,LVedv,LVesv,LVef) in enumerate(dataset):
        if gpu:
            patch1, patch2, patch11, patch22 = img1.cuda(), img2.cuda(), img11.cuda(), img22.cuda()
            true_masks1, true_masks2, true_masks11, true_masks22 = mask1.float().cuda(), mask2.float().cuda(), mask11.float().cuda(), mask22.float().cuda()
            LVedv, LVesv, LVef = (LVedv*0.001).float().cuda(),(LVesv*0.001).float().cuda(),(LVef*0.01).float().cuda()
            #dist_m1, dist_m2, dist_m11, dist_m22 = boundary1.cuda(), boundary2.cuda(), boundary11.cuda(), boundary22.cuda()
        with torch.no_grad():
            mask_pred1,mask_pred2,mask_pred3,mask_pred4,_,_,_,_, Cal_out1,Cal_out2, Cal_out3=net.forward(patch1, patch2, patch11, patch22)
        #mask_pred1 ,mask_pred2 ,mask_pred3 ,mask_pred4 = (mask_pred1 > 0.5).float(), (mask_pred2 > 0.5).float(),(mask_pred3 > 0.5).float(), (mask_pred4 > 0.5).float()

        mask_pred1 ,mask_pred2 ,mask_pred3 ,mask_pred4 = (mask_pred1).float(), (mask_pred2 ).float(),(mask_pred3 ).float(), (mask_pred4).float()
        save_path= 'nest_dual3/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_image(torch.argmax(F.softmax(mask_pred1,dim=1), 1).float().data.cpu(), save_path + "/" + str(i) + "_" + 'ed.png')
        save_image(torch.argmax(F.softmax(mask_pred2,dim=1),1).float().data.cpu(), save_path + "/" + str(i) + "_" + 'es.png')
        save_image(torch.argmax(F.softmax(mask_pred3,dim=1),1).float().data.cpu(), save_path + "/" + str(i) + "_" + 'ed2.png')
        save_image(torch.argmax(F.softmax(mask_pred4,dim=1),1).float().data.cpu(), save_path + "/" + str(i) + "_" + 'es2.png')
        #s_d1= _get_segmentation_cost(mask_pred1, true_masks1).item()+_get_segmentation_cost(mask_pred2, true_masks2).item()#dice_coeff(mask_pred1, true_masks1).item()+dice_coeff(mask_pred2, true_masks2).item()
        #s_d2 = _get_segmentation_cost(mask_pred3, true_masks11).item()+_get_segmentation_cost(mask_pred4, true_masks22).item()
        s_d1= _eval_dice(mask_pred1, true_masks1).item()+_eval_dice(mask_pred2, true_masks2).item()#dice_coeff(mask_pred1, true_masks1).item()+dice_coeff(mask_pred2, true_masks2).item()
        s_d2 = _eval_dice(mask_pred3, true_masks11).item()+_eval_dice(mask_pred4, true_masks22).item()
        tot1=tot1 +s_d1
        tot2=tot2 +s_d2
        err_edv = err_edv+torch.abs(Cal_out1.float()-LVedv)
        err_esv = err_esv+torch.abs(Cal_out2.float()-LVesv)
        err_ef = err_ef+torch.abs(Cal_out3.float()-LVef)
    return tot1 / (i + 1)/2, tot2 / (i + 1)/2,err_edv.item(),err_esv.item(),err_ef.item()

print('==> Preparing data..')
'''
transform = transforms.Compose([transforms.Resize(224),
    transforms.ToTensor()
])
'''
transform = transforms.Compose([#transforms.CenterCrop(550),
                                #transforms.RandomResizedCrop(210, scale=(0.9, 1.1), ratio=(1,1), interpolation=2),
    #transforms.RandomRotation(5),
    transforms.Resize(112),
    transforms.ToTensor()
])
transform1 = transforms.Compose([#transforms.Resize(224),
    transforms.Resize(112),
    transforms.ToTensor()
])
for k in range(1):
    trainset = ListDataset(root=['gt_img', 'index_img'],
                           list_file='all_info.txt', state='Train',input_size=112, transform=transform, k=k)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, num_workers=4)

    testset = ListDataset(root=['gt_img', 'index_img'],
                          list_file='all_info.txt', state='Valid',input_size=112, transform=transform1, k=k)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)
    net = MyNet(input_channels=1, num_classes=4,beta=0.5)
    dir_checkpoint = 'a_task6_1_final/'
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)
    #net.load_state_dict(torch.load('a_task4/' + 'model.pth'),strict=False)
    #net=nn.DataParallel(net)
    net = net.cuda()
    net = nn.DataParallel(net, device_ids=[0,1])
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
    mx_val = 0
    epochs = 200
    warm_up_epochs=10
    warm_up_with_cosine_lr = lambda epoch: epoch / warm_up_epochs if epoch <= warm_up_epochs else 0.5 * (
            math.cos((epoch - warm_up_epochs) / (epochs - warm_up_epochs) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    for epoch in range(epochs):
        for step, (img1, img2, mask1,mask2,img11, img22, mask11, mask22,c1,c2,c11,c22,LVedv,LVesv,LVef) in enumerate(trainloader):
            patch1, patch2 ,patch11, patch22 = img1.cuda(), img2.cuda(), img11.cuda(), img22.cuda()
            c1, c2, c11, c22=c1.cuda(),c2.cuda(),c11.cuda(),c22.cuda()
            true_masks1,true_masks2 ,true_masks11,true_masks22 = mask1.float().cuda(), mask2.float().cuda(),mask11.float().cuda(), mask22.float().cuda()
            #dist_m1,dist_m2,dist_m11,dist_m22=boundary1.cuda(),boundary2.cuda(),boundary11.cuda(),boundary22.cuda()
            LVedv, LVesv, LVef = (LVedv*0.001).float().cuda(),(LVesv*0.001).float().cuda(),(LVef*0.01).float().cuda()
            #ed_seg , es_seg , ed_seg2 , es_seg2 ,ed_c, es_c , ed_c2 , es_c2=net.forward(patch1, patch2 ,patch11, patch22)
            ed_seg , es_seg , ed_seg2 , es_seg2 ,ed_c, es_c , ed_c2 , es_c2, Cal_out1,Cal_out2, Cal_out3=net(patch1, patch2 ,patch11, patch22)

            #if isinstance(net, torch.nn.DataParallel):
            #    net = net.module
            elbo = elbo_loss(true_masks1,true_masks2,true_masks11,true_masks22,c1, c2, c11,c22,ed_seg , es_seg , ed_seg2 , es_seg2 ,ed_c, es_c , ed_c2 , es_c2, Cal_out1,Cal_out2, Cal_out3,LVedv, LVesv, LVef)
            reg_loss = l2_regularisation(net)
            loss = 1e-5 * reg_loss + elbo
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        print('Epoch {} finished !'.format(epoch + 1),
              'Loss: {}'.format(loss / (step + 1)))  # ,'EncLoss: {}'.format(enc_loss / i)
        if 1:
            val_dice1, val_dice2, edv_val, esv_val, ef_val  = eval_gan_nest(net, testloader, gpu=True)
            val_dice=(val_dice1+val_dice2)/2
            print(
                'Validation Dice Coeff: {} :{},{},Index error:LVedv: {},LVesv: {},LVef: {}'.format(val_dice, val_dice1,
                                                                                                   val_dice2, edv_val,
                                                                                                   esv_val, ef_val))

        if (val_dice > mx_val)  :
            mx_val=val_dice
            torch.save(net.state_dict(),dir_checkpoint + 'model.pth'.format(k,epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))