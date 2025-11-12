import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import cv2 as cv
import numpy as np

def normalize_with_limit(x, limit=1000.0, eps=1e-5):
    """归一化并限制最大值"""
    max_val = torch.max(x)
    max_val = torch.clamp(max_val, max=max_val.new_tensor(limit))
    return x / (max_val + eps)

def getBAV(label, num_class):
    """分离每类二值掩码"""
    bav = torch.zeros((num_class, label.shape[0], label.shape[1]), device=label.device)
    for i in range(num_class):
        mask = (label == i).float()
        bav[i] = mask
    return bav

def getDW(lab_bav, pred_bav, e=1e-5, limit=1000.0):
    """计算距离权重并归一化"""
    for i in range(lab_bav.shape[0]):
        lab_bav[i] = torch.tensor(distance_transform_edt(lab_bav[i].cpu().numpy()), device=lab_bav.device)
        lab_bav[i] = normalize_with_limit(lab_bav[i], limit, e)

    for i in range(pred_bav.shape[0]):
        pred_bav[i] = torch.tensor(distance_transform_edt(pred_bav[i].cpu().numpy()), device=pred_bav.device)
        pred_bav[i] = normalize_with_limit(pred_bav[i], limit, e)

    return lab_bav, pred_bav

class VE_Loss(nn.Module):
    def __init__(self, num_class=3, limit=576*576):
        super(VE_Loss, self).__init__()
        self.num_class = num_class
        self.limit = limit

    def forward(self, pred, lab):
        pred_softmax = torch.argmax(pred, dim=1)   # [B,H,W]
        # 确保输入预测标签的维度一致性
        if pred_softmax.shape!=lab.shape:
            lab = lab.reshape(pred_softmax.shape)
        loss_sum = 0.0
        # 逐个batch计算
        for i in range(pred_softmax.shape[0]):
            # 分离动静脉 (注意：lab_bav ← lab, pred_bav ← pred)
            lab_bav = getBAV(lab[i], self.num_class)
            pred_bav = getBAV(pred_softmax[i], self.num_class)

            # 距离权重
            lab_bav, pred_bav = getDW(lab_bav, pred_bav, limit=self.limit)

            # one-hot 编码
            lab_one_hot = F.one_hot(lab[i].to(torch.int64), self.num_class).permute(2,0,1).float()
            pred_one_hot = F.one_hot(pred_softmax[i].to(torch.int64), self.num_class).permute(2,0,1).float()

            # 特征拼接
            lab_feat = torch.cat((torch.sum(lab_bav, dim=0, keepdim=True), lab_one_hot), dim=0)
            pred_feat = torch.cat((torch.sum(pred_bav, dim=0, keepdim=True), pred_one_hot), dim=0)

            # 只取非背景像素
            mask = (lab[i] > 0)
            if mask.sum() == 0:   # 跳过空标签
                continue

            # 拼接向量
            # 只算目标血管区域
            # pred_result = pred_feat[:, mask].permute(1,0)  # [N, C]
            # lab_result  = lab_feat[:, mask].permute(1,0)   # [N, C]
            # 即算目标血管区域，又算背景区域
            pred_result = pred_feat.reshape(4,-1).permute(1,0)  # [N, C]
            lab_result  = lab_feat.reshape(4,-1).permute(1,0)   # [N, C]

            # 计算余弦相似度
            sim = torch.cosine_similarity(pred_result, lab_result, dim=-1, eps=1e-8)
            loss_value = 1 - sim.mean()

            # 累加
            loss_sum = loss_sum + loss_value
        return torch.nan_to_num(loss_sum / pred_softmax.shape[0], nan=0.0, posinf=1.0, neginf=0.0)   # 归一化 batch 大小

if __name__ == "__main__":
    loss = VE_Loss()
    pred = torch.cat([torch.zeros(4,1,576,576),torch.zeros(4,1,576,576),torch.zeros(4,1,576,576)],dim=1)
    lab = cv.imread("datasets/DRIVE_AV/test/labels/01_test.png",cv.IMREAD_GRAYSCALE)
    lab = cv.resize(lab,(576,576),interpolation=cv.INTER_NEAREST)
    lab[lab==3]=2
    lab = torch.tensor(lab,dtype=torch.long).reshape(1,1,576,576)
    lab = torch.cat([lab,lab,lab,lab],dim=0).unsqueeze(1)
    pred = pred.to("cuda:0")
    lab = lab.to("cuda:0")
    l = loss(pred,lab)
    print(l)