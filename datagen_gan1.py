import os
import pickle
import torch
import numpy as np
import random
import sys
import random
from torch.utils import data
import torchvision.transforms as transforms
from scipy import ndimage
from skimage import data,filters,feature
from scipy.ndimage import distance_transform_edt as distance
from PIL import Image, ImageOps, ImageFilter

import matplotlib.pyplot as plt

# from transform1 import randonm_resize, random_rotate, rotate_resize


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, root, list_file, state,input_size, transform, k):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.root = root
        self.state = state
        self.transform = transform
        self.fnames = []
        self.input_size=input_size

        # for line in lines:
        #    self.fnames.append(line[:-1])

        if self.state == 'Train':
            with open(list_file) as f:
                lines = f.readlines()
            len_line = len(lines)
            self.fnames.extend(lines[:int((k % 10) * len_line / 10)])
            self.fnames.extend(lines[int((k % 10 + 1) * len_line / 10):-1])
            self.num_samples = len(self.fnames)

        if self.state == 'Valid':
            with open(list_file) as f:
                lines = f.readlines()
            len_line = len(lines)
            a=int((k % 10) * len_line / 10)+1
            b=int((k % 10 + 1) * len_line / 10)
            self.fnames.extend(lines[int((k % 10) * len_line / 10)+1:int((k % 10 + 1) * len_line / 10)])
            self.num_samples = len(self.fnames)

        if self.state == 'test':
            with open(list_file) as f:
                lines = f.readlines()
            self.fnames.extend(lines)
            self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.

        idx = idx % self.num_samples
        fname1 = self.fnames[idx].split(' ')[0] + '_4CH_ED.png'
        mask_fname1 = self.fnames[idx].split(' ')[0] + '_4CH_ED_gt.png'
        img1 = Image.open(os.path.join(self.root[0], fname1))
        img1 = self.resize_image__(img1)
        mask1 = Image.open(os.path.join(self.root[1], mask_fname1))
        mask1 = self.resize_image__(mask1)
        ##################################################
        fname2 = self.fnames[idx].split(' ')[0] + '_4CH_ES.png'
        mask_fname2  = self.fnames[idx].split(' ')[0] + '_4CH_ES_gt.png'
        img2 = Image.open(os.path.join(self.root[0], fname2))
        img2 = self.resize_image__(img2)
        mask2  = Image.open(os.path.join(self.root[1], mask_fname2))
        mask2  = self.resize_image__(mask2)
        ############################################################
        fname11 = self.fnames[idx].split(' ')[0] + '_2CH_ED.png'
        mask_fname11 = self.fnames[idx].split(' ')[0] + '_2CH_ED_gt.png'
        img11 = Image.open(os.path.join(self.root[0], fname11))
        img11 = self.resize_image__(img11)
        mask11 = Image.open(os.path.join(self.root[1], mask_fname11))
        mask11 = self.resize_image__(mask11)
        ##################################################
        fname22 = self.fnames[idx].split(' ')[0] + '_2CH_ES.png'
        mask_fname22 = self.fnames[idx].split(' ')[0] + '_2CH_ES_gt.png'
        img22 = Image.open(os.path.join(self.root[0], fname22))
        img22 = self.resize_image__(img22)
        mask22 = Image.open(os.path.join(self.root[1], mask_fname22))
        mask22 = self.resize_image__(mask22)
        ##############################################################
        LVedv = self.fnames[idx].split('  ')[1]
        LVesv = self.fnames[idx].split('  ')[2]
        LVef = self.fnames[idx].split('  ')[3][:-1]
        #######################################################################
        if self.state == 'Train':
            img1, mask1,c1= self.transform_mul_mask(img1, mask1)
            img2, mask2 ,c2= self.transform_mul_mask(img2, mask2)
            img11, mask11,c11= self.transform_mul_mask(img11, mask11)
            img22, mask22,c22 = self.transform_mul_mask(img22, mask22)
            LVedv ,LVesv, LVef = self._index_transform(LVedv), self._index_transform(LVesv), self._index_transform(LVef)
        else:
            img1, mask1 = self.transform_mul_mask_val(img1, mask1)
            img2, mask2  = self.transform_mul_mask_val(img2, mask2)
            img11, mask11 = self.transform_mul_mask_val(img11, mask11)
            img22, mask22  = self.transform_mul_mask_val(img22, mask22)
            LVedv, LVesv, LVef = self._index_transform(LVedv), self._index_transform(LVesv), self._index_transform(LVef)

        #boundary = self.dist_image__(mask)
        if self.transform is not None:
            img1 ,img2= self.transform(img1),self.transform(img2)
            #mask1,mask2 = torch.stack([mask1], dim=0),torch.stack([mask2], dim=0)#mask = torch.stack([mask], dim=0)
            img11, img22 = self.transform(img11), self.transform(img22)
            #mask11, mask22 = torch.stack([mask11], dim=0) ,torch.stack([mask22], dim=0) # mask = torch.stack([mask], dim=0)
            LVedv = torch.stack([LVedv], dim=0)
            LVesv = torch.stack([LVesv], dim=0)
            LVef = torch.stack([LVef], dim=0)
            #print('-->',img.shape, mask.shape)
           # print(img.shape, mask.shape)
        if self.state == 'Train':
            return img1, img2, mask1,mask2,img11, img22, mask11, mask22,c1,c2,c11,c22,LVedv,LVesv,LVef
        else:
            return img1, img2, mask1,mask2,img11, img22, mask11, mask22,LVedv,LVesv,LVef



    def resize_image__(self, image):
        top, bottom, left, right = (0, 0, 0, 0)

        # 获取图像尺寸

        w, h = image.size

        # 对于长宽不相等的图片，找到最长的一边
        longest_edge = max(h, w)

        # 计算短边需要增加多上像素宽度使其与长边等长
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass

            # RGB颜色
        if image.mode == 'RGB':
            cv_value = (0, 0, 0)
        else:
            cv_value = 0
        newImg = Image.new(image.mode, (h + top + bottom, w + left + right), cv_value)

        # 给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
        box = (left, top, left + w, top + h)
        # 设置要裁剪的区域
        # region = newImg.crop(box)  # 此时，region是一个新的图像对象。im_crop = im.crop(box)
        newImg.paste(image, box)
        # newImg = newImg.resize((224, 224), Image.BILINEAR)
        return newImg


    def _label_decomp(self, mask):
        """
                decompose label for softmax classifier
                original labels are batchsize * W * H * 1, with label values 0,1,2,3...
                this function decompse it to one hot, e.g.: 0,0,0,1,0,0 in channel dimension
                numpy version of tf.one_hot
                """
        mask=np.array(mask)
        num_cls = 4
        one_hot = []
        for i in range(num_cls):
            _vol = np.zeros(mask.shape)
            _vol[mask == i] = 1.0
            one_hot.append(_vol)
        return np.stack(one_hot, axis=0)

    def transform_mul_mask(self, img, mask):
        # random mirror
        angle=random.randint(0,10)
        img=img.rotate(angle)
        mask = mask.rotate(angle)
        mask = mask.resize((self.input_size, self.input_size), Image.NEAREST)
        label_map = self._label_decomp(mask)
        coutour_group=self._get_coutour_sample(label_map)
        # final transform
        return img,torch.LongTensor(label_map.astype('float32')),torch.LongTensor(coutour_group.astype('float32'))  #

    def transform_mul_mask_val(self, img, mask):
        #mask = mask.resize((self.input_size, self.input_size), Image.BILINEAR)
        #boundary = self.dist_image__(mask)
        mask = mask.resize((self.input_size, self.input_size), Image.NEAREST)
        label_map = self._label_decomp(mask)
        #plt.imshow(mask)
        #plt.show()
        # final transform
        return img,torch.LongTensor(label_map.astype('float32'))


    def _img_transform(self, img):
        return np.array(img)

    def _get_coutour_sample1(self,y_true):
        """
        y_true: BxHxWx2
        """
        positive_mask = np.expand_dims(y_true[..., 1], axis=3)
        metrix_label_group = np.expand_dims(np.array([1, 0, 1, 1, 0]), axis=1)
        coutour_group = np.zeros(positive_mask.shape)

        for i in range(positive_mask.shape[0]):
            slice_i = positive_mask[i]

            if metrix_label_group[i] == 1:
                # generate coutour mask
                erosion = ndimage.binary_erosion(slice_i[..., 0], iterations=1).astype(slice_i.dtype)
                sample = np.expand_dims(slice_i[..., 0] - erosion, axis=2)

            elif metrix_label_group[i] == 0:
                # generate background mask
                dilation = ndimage.binary_dilation(slice_i, iterations=5).astype(slice_i.dtype)
                sample = dilation - slice_i

            coutour_group[i] = sample
        return coutour_group, metrix_label_group

    def _get_coutour_sample(self,y_true):
        """
        y_true: BxnxHxW
        """
        coutour_group = []
        for i in range(y_true.shape[0]):
            slice_i = y_true[i, :, :]
            if i == 0:
                slice_i = 1 - slice_i
            dilation = ndimage.binary_dilation(slice_i, iterations=2).astype(slice_i.dtype)
            # sample = dilation - slice_i
            erosion = ndimage.binary_erosion(slice_i, iterations=2).astype(slice_i.dtype)
            sample = dilation - erosion
            coutour_group.append(sample)
        return np.stack(coutour_group, axis=0)

    def _index_transform(self, index):
        return torch.LongTensor(np.array(index).astype('float32'))#

    def __len__(self):
        return self.num_samples




