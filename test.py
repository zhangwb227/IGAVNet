# -*- coding:utf-8 -*-
import os, argparse
from tqdm import tqdm
import torch
import numpy as np
import cv2 as cv
from loaddata.LoadDatasets import LoadDatasets
from models.IVGNet import IVGNet
from util.utils import DiceLoss,cal_miou,BinaryDiceLoss
from util.log_function import print_options, print_network
from scipy.ndimage import zoom
# 损失函数对比实验
from torchvision.utils import save_image

""" set flags / seeds """
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
"""
求解最佳pth路径
"""
def getBestPthPath(results_path):
    pth_list = os.listdir(results_path)
    pth_dict = {}
    for pth in pth_list:
        pth_dict[pth[pth.find(".")+1:pth.rfind(".")]]=os.path.join(results_path,pth)
    pth_dict_key = sorted(pth_dict.keys())
    return pth_dict[pth_dict_key[-1]]

if __name__ == "__main__":
    """ Hpyer parameters """
    parser = argparse.ArgumentParser(description="")
    # training option
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--model_name', type=str, default="IGAVNet")
    # data option
    parser.add_argument('--test_img_path', type=str, default="datasets/HRF_AV/test/images/")
    parser.add_argument('--test_lab_path', type=str, default="datasets/HRF_AV/test/labels/")
    parser.add_argument('--result_dir', type=str, default='results/HRF_AV/IGAVNet_clDice_HRF_AV/')
    parser.add_argument('--pth_path', type=str, default='results/HRF_AV/IGAVNet/xxx.pth')
    # DRIVE_AV;HRF_AV;LES_AV
    parser.add_argument('--dataset_name', type=str,default="HRF_AV")
    opt = parser.parse_args()

    if not os.path.exists(os.path.join(opt.result_dir)):
        os.mkdir(os.path.join(opt.result_dir))

    if not os.path.exists(os.path.join(opt.result_dir,"test_result")):
        os.mkdir(os.path.join(opt.result_dir,"test_result"))
    if not os.path.exists(os.path.join(opt.result_dir,"test_shows")):
        os.mkdir(os.path.join(opt.result_dir,"test_shows"))
    if not os.path.exists(os.path.join(opt.result_dir,"soft_shows")):
        os.mkdir(os.path.join(opt.result_dir,"soft_shows"))

    for i in os.listdir(os.path.join(opt.result_dir,"test_result")):
        os.remove(os.path.join(opt.result_dir,"test_result",i))
        os.remove(os.path.join(opt.result_dir,"soft_shows",i))
        os.remove(os.path.join(opt.result_dir,"test_shows",i))
    """ dataset and dataloader """
    train_dataset = LoadDatasets(img_path=opt.test_img_path,
                                 lab_path=opt.test_lab_path,
                                 dataset_name=opt.dataset_name,
                                 is_aug=False)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=False,
                                             num_workers=4)
    """ device configuration """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    # select model name
    model = IVGNet(in_size=3,out_size=128,n_classes=3,
                        middle_size=128,device=device,
                        block_size_w=36,block_size_h=36)

    """model init or load checkpoint"""
    model.load_state_dict(torch.load(opt.pth_path))
    """ training part """
    model = model.to(device)
    with tqdm(total= len(train_loader)) as _tqdm:
        _tqdm.set_description('epoch: {}/{}'.format(1, 1))
        model.eval()
        for j, (img, lab,img_name) in enumerate(train_loader):
            img = img.to(device)
            out = model(img)
            """
            存储png结果
            包括可视化结果和二值结果
            """
            pred = torch.argmax(out,dim=1).detach().cpu().numpy()
            r = pred[0].copy()
            g = pred[0].copy()
            b = pred[0].copy()
            r[r!=1]=0
            g[g!=0]=0
            b[b!=2]=0
            b[b==2]=1
            result = cv.merge([g*255,r*255,b*255])
            cv.imwrite(os.path.join(opt.result_dir,"test_result/",img_name[0]), np.uint8(pred[0]))
            cv.imwrite(os.path.join(opt.result_dir,"test_shows/",img_name[0]),np.uint8(result))
            pred_softmax = torch.nn.functional.softmax(out, dim=1)
            pred_softmax = pred_softmax.cpu().detach().numpy()[0]
            r,g,b = pred_softmax[0], pred_softmax[1], pred_softmax[2]
            B=0*r+0*g+255*b
            G=0*r+255*g+0*b
            R=0*r+0*g+0*b
            soft_result = cv.merge([R,G,B])
            cv.imwrite(os.path.join(opt.result_dir,"soft_shows/",img_name[0]),np.uint8(soft_result))
            _tqdm.update(1)
