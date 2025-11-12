# -*- coding:utf-8 -*-
import os
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import imgaug.augmenters as iaa
# import cv2 as cv
import torch
from PIL import Image
from torchvision.utils import save_image

class LoadDatasets(Dataset):
    def __init__(self, img_path,lab_path,dataset_name="DRIVE_AV",is_aug=True):
        self.img_paths = []
        self.lab_paths = []
        self.img_names = []
        self.dataset_name=dataset_name
        self.TF = transforms.Compose([
            transforms.ToTensor()
        ])
        self.is_aug = is_aug
        # 数据增强，随机选择0-3个增强模式
        self.seq1 = iaa.SomeOf((0,1),[
                    iaa.Affine(rotate=(-90, 90), mode="constant"),  # 旋转
                    iaa.Fliplr(),  # 水平翻转
                    iaa.Flipud(),  # 垂直翻转
                    ],random_order=True)
        self.seq2 = iaa.SomeOf((0,2),[
                    iaa.GaussianBlur(sigma=(0, 1.5)),
                    iaa.Sharpen(alpha=(0, 0.3), lightness=(0.9, 1.1)),
                    iaa.LinearContrast((0.75, 1.5), per_channel=True),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                    iaa.Multiply((0.8, 1.2), per_channel=0.2),
                    ],random_order=True)
        for image_name in os.listdir(img_path):
            self.img_paths.append(os.path.join(img_path,image_name))
            self.img_names.append(os.path.join(image_name))
            self.lab_paths.append(os.path.join(lab_path,image_name))
    
    def __getitem__(self, index):
        W,H = 576,576
        img = Image.open(self.img_paths[index])
        lab = Image.open(self.lab_paths[index]).convert('L')
        img = self.resize_and_pad(img, target_size=(W, H))
        lab = self.resize_and_pad(lab, target_size=(W, H),is_mask=True)
        img = np.array(img)
        lab = np.array(lab)
        if self.dataset_name=="DRIVE_AV":
            # DRIVE交叉点赋值为静脉
            lab[lab==3]=2
        elif self.dataset_name=="LES_AV":
            # DRIVE交叉点赋值为静脉
            lab[lab==3]=2
        elif self.dataset_name=="HRF_AV":
            # DRIVE交叉点赋值为静脉
            lab[lab==3]=2

        if self.is_aug:
            lab = lab.reshape(W,H,1)
            img_lab = np.concatenate([img, lab], axis=2)
            # 图像和标签同时变换
            img_lab = self.seq1(image=img_lab)
            img = img_lab[:,:,0:3]
            lab = img_lab[:,:,3:4].reshape(W,H)
            # 图像增强
            img = self.seq2(image=img)
        
        img = self.TF(img.copy())
        # transforms1转为0-1的浮点数，标签强度减弱
        lab = torch.from_numpy(np.ascontiguousarray(lab.copy()).astype(np.float32))
        return img, lab, self.img_names[index]

    def __len__(self):
        return len(self.img_paths)
    
    def resize_and_pad(self, image, target_size=(576, 576), is_mask=False):
        """
        将图像等比缩放并填充到指定大小

        参数:
            image: PIL.Image 对象（图像或 mask）
            target_size: 输出尺寸，如 (786, 512)
            is_mask: 是否是分割标签，决定插值方式

        返回:
            result: 目标大小的 PIL.Image 对象
        """
        target_w, target_h = target_size
        w, h = image.size
        # 等比缩放
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        # 选择插值方式
        resample = Image.NEAREST if is_mask else Image.BILINEAR
        resized = image.resize((new_w, new_h), resample=resample)
        # 创建空图（黑色填充）
        if is_mask:
            new_image = Image.new("L", (target_w, target_h), 0)  # mask：单通道
        else:
            new_image = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        # 将 resized 图像粘贴到中央
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        new_image.paste(resized, (paste_x, paste_y))
        return new_image
        
        
  
if __name__ == "__main__":

    img_path = "datasets/LES_AV/train/images/"
    lab_path = "datasets/LES_AV/train/labels/"
    train_dataset = LoadDatasets(img_path=img_path,lab_path=lab_path,dataset_name="LES_AV",is_aug=True)
    print("数据个数：", train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
    print(len(train_loader))
    for batch, [img, lab, name] in enumerate(train_loader):
        print(batch, len(name),img.shape,torch.max(lab),torch.min(lab),torch.unique(lab))
        # print(img.shape,lab.shape)
        result = torch.cat([img,lab.unsqueeze(1)],dim=1).view(-1,1,576,576)
        save_image(result,"test.png",nrow=4)
        # img = img[0].permute(1,2,0).detach().cpu().numpy()*255
        # cv.imwrite("img.png",np.uint8(img))
        # lab = lab[0].detach().cpu().numpy()*100
        # cv.imwrite("lab.png",np.uint8(lab))
        break


