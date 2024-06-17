import os
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import re
import random
import rawpy
import PIL.Image
import sys
sys.path.append("../.")
sys.path.append("../data/")
sys.path.append("..")
# import helper



class Data:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loaders(self, parse_patches=True):

        train_path = os.path.join(self.config.data.train_data_dir)
        val_path = os.path.join(self.config.data.test_data_dir)

        train_dataset = MyDataset(train_path,
                                  n=self.config.training.patch_n,
                                  patch_size=self.config.data.image_size,
                                  transforms=self.transforms,
                                  parse_patches=parse_patches)
        val_dataset = MyDataset(val_path,
                                n=self.config.training.patch_n,
                                patch_size=self.config.data.image_size,
                                transforms=self.transforms,
                                parse_patches=parse_patches)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        # 训练数据
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=False)
        # 评估数据
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=False)

        return train_loader, val_loader

def pack_raw_bayer(raw, wp=1023, clip=True):
    #pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    raw_pattern = raw.raw_pattern
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    B = np.where(raw_pattern==2)
    G2 = np.where(raw_pattern==3)
    
    white_point = wp
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.stack((im[R[0][0]:H:2, R[1][0]:W:2], #RGBG
                    im[G1[0][0]:H:2, G1[1][0]:W:2],
                    im[B[0][0]:H:2, B[1][0]:W:2],
                    im[G2[0][0]:H:2, G2[1][0]:W:2]), axis=0).astype(np.float32)

    black_level = np.array(raw.black_level_per_channel)[:,None,None].astype(np.float32)

    out = (out - black_level) / (white_point - black_level)
    out = np.clip(out, 0.0, 1.0) if clip else out
    
    return out

def get_data_path(filename):
    with open(filename) as f:
        fns = f.readlines()
        fns = [tuple(fn.strip().split(' '))for fn in fns]
    input = []
    gt = []
    for i in range(1864):
        input.append(fns[i][0])
        gt.append(fns[i][1])
    return input, gt

# 数据集加载类
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, n, transforms, parse_patches=True):
        super().__init__()

        self.dir = dir
        # input_names = os.listdir(dir+'short')
        # gt_names = os.listdir(dir+'long')
        
        self.input_names, self.gt_names = get_data_path("/raid/qinjiahao/data/Sony_train_list.txt")
        
        # self.input_names = input_names
        # self.gt_names = gt_names
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches

    @staticmethod
    def get_params(img, output_size, n):
        ch, w, h = img.shape
        # print("w: {}, h: {}".format(w, h))
        th, tw = output_size
        # print("tw: {}, th: {}".format(tw, th))
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        # print("x {}".format(len(x)))
        for i in range(len(x)):
            # 自己添加的格式转换
            new_img = img[:, y[i]:y[i]+h, x[i]:x[i]+w]
            # img = PIL.Image.fromarray(np.uint8(img))
            
            # new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_img)
        return tuple(crops)

    def get_images(self, index):
        input_name = self.input_names[index]
        img_id = re.split('/', input_name)[-1][:-4]
        gt_name = self.gt_names[index]
        input_img = rawpy.imread("/raid/qinjiahao/data/" + input_name)
        input_img = pack_raw_bayer(input_img)
        # helper.logging.save_image(input_img, os.path.join("/raid/qinjiahao/data/Sony/result/", str(index), f"{index}_origin.png"))
        
        try:
            gt_img = rawpy.imread("/raid/qinjiahao/data/" + gt_name)
            gt_img = pack_raw_bayer(gt_img)
        except:
            # rawpy.imread(gt_name).convert('RGB')
            gt_img = rawpy.imread("/raid/qinjiahao/data/" + gt_name)
            gt_img = pack_raw_bayer(gt_img)

        if self.parse_patches:
            i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n)
            input_img = self.n_random_crops(input_img, i, j, h, w)
            gt_img = self.n_random_crops(gt_img, i, j, h, w)
            outputs = [torch.cat([self.transforms(input_img[i]), self.transforms(gt_img[i])], dim=1)
                       for i in range(self.n)]
            return torch.stack(outputs, dim=0), img_id
        else:
            # Resizing images to multiples of 16 for whole-image restoration
            print(input_img.shape)
            ch, wd_new, ht_new = input_img.shape
            if ht_new > wd_new and ht_new > 1024:
                wd_new = int(np.ceil(wd_new * 1024 / ht_new))
                ht_new = 1024
            elif ht_new <= wd_new and wd_new > 1024:
                ht_new = int(np.ceil(ht_new * 1024 / wd_new))
                wd_new = 1024
            wd_new = int(16 * np.ceil(wd_new / 16.0))
            ht_new = int(16 * np.ceil(ht_new / 16.0))
            input_img = input_img[:, 0:wd_new, 0:ht_new]
            gt_img = gt_img[:, 0:wd_new, 0:ht_new]
            # input_img = input_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
            # gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)

            return torch.cat([self.transforms(input_img), self.transforms(gt_img)], dim=1), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
