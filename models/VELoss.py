import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv



def gaussian_blur(x, kernel_size=7, sigma=2):
    C, H, W = x.shape

    grid = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - kernel_size // 2
    gaussian = torch.exp(-(grid**2) / (2 * sigma**2))
    kernel = gaussian[:, None] @ gaussian[None, :]
    kernel = kernel / kernel.sum()

    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(C, 1, 1, 1)

    return F.conv2d(x, kernel, padding=kernel_size//2, groups=C)

def soft_distance_map(x, kernel_size=7):
    padding = kernel_size // 2
    dist = F.avg_pool2d(x, kernel_size, stride=1, padding=padding)
    return dist


class VE_Loss(nn.Module):
    def __init__(self, num_class=3, kernel_size=9):
        super(VE_Loss, self).__init__()
        self.num_class = num_class
        self.kernel_size = kernel_size

    def forward(self, pred, lab):
        """
        pred: [B, C, H, W] (logits)
        lab:  [B, H, W]
        """

        B, C, H, W = pred.shape

        pred_prob = F.softmax(pred, dim=1)

        lab_one_hot = F.one_hot(lab.long(), self.num_class).permute(0,3,1,2).float()

        loss_sum = 0.0

        for i in range(B):

            # pred_dist = soft_distance_map(pred_prob[i], self.kernel_size)
            # lab_dist  = soft_distance_map(lab_one_hot[i], self.kernel_size)

            pred_dist = gaussian_blur(pred_prob[i], self.kernel_size)
            lab_dist  = gaussian_blur(lab_one_hot[i], self.kernel_size)
            

            pred_dist_sum = torch.sum(pred_dist, dim=0, keepdim=True)
            lab_dist_sum  = torch.sum(lab_dist, dim=0, keepdim=True)

            pred_feat = torch.cat((pred_dist_sum, pred_prob[i]), dim=0)
            lab_feat  = torch.cat((lab_dist_sum, lab_one_hot[i]), dim=0)

            pred_vec = pred_feat.view(pred_feat.shape[0], -1).permute(1,0)
            lab_vec  = lab_feat.view(lab_feat.shape[0], -1).permute(1,0)

            sim = F.cosine_similarity(pred_vec, lab_vec, dim=-1, eps=1e-8)

            loss = 1 - sim.mean()
            loss_sum += loss

        return loss_sum / B
    

if __name__ == "__main__":
    loss = VE_Loss()
    pred = torch.abs(torch.cat([torch.randn(4,1,576,576),torch.randn(4,1,576,576),torch.randn(4,1,576,576)],dim=1))
    lab = cv.imread("datasets/DRIVE_AV/test/labels/01_test.png",cv.IMREAD_GRAYSCALE)
    lab = cv.resize(lab,(576,576),interpolation=cv.INTER_NEAREST)
    lab[lab==3]=2
    lab = torch.tensor(lab,dtype=torch.long).reshape(1,576,576)
    lab = torch.cat([lab,lab,lab,lab],dim=0)
    pred = pred.to("cuda:0")
    lab = lab.to("cuda:0")
    print(pred.shape,lab.shape)
    l = loss(pred,lab)
    print(l)