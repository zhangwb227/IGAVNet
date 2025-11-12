# -*- coding:utf-8 -*-
import os, argparse
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
from loaddata.LoadDatasets import LoadDatasets
from models.IVGNet import IVGNet
from util.utils import DiceLoss,cal_miou
from models.VELoss import VE_Loss
from models.loss_CE import FocalTverskyLoss
from util.SaveModelPth import CheckpointManager
from util.log_function import print_options, print_network
# 损失函数对比实验
from torchvision.utils import save_image

if __name__ == "__main__":
    """ Hpyer parameters """
    parser = argparse.ArgumentParser(description="")
    # training option
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--Epochs', type=int, default=60)
    parser.add_argument('--numbers', type=int, default=12)
    parser.add_argument('--model_name', type=str, default="IVGNet")
    parser.add_argument('--loss_name', type=str, default="CE_DSC_FT_VE")
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--train_img_path', type=str, default="datasets/HRF_AV/train/images/")
    parser.add_argument('--train_lab_path', type=str, default="datasets/HRF_AV/train/labels/")
    parser.add_argument('--val_img_path', type=str, default="datasets/HRF_AV/val/images/")
    parser.add_argument('--val_lab_path', type=str, default="datasets/HRF_AV/val/labels/")
    parser.add_argument('--result_dir', type=str,default='results/IVGNet/')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default="0")
    # DRIVE_AV;HRF_AV;LES_AV
    parser.add_argument('--dataset_name', type=str,default="HRF_AV")
    opt = parser.parse_args()
    """ set flags / seeds """
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.CUDA_VISIBLE_DEVICES

    if not os.path.exists(opt.result_dir): os.mkdir(opt.result_dir)
    if not os.path.exists(os.path.join(opt.result_dir,"result")):
        os.mkdir(os.path.join(opt.result_dir,"result"))
    if not os.path.exists(os.path.join(opt.result_dir,"pth")):
        os.mkdir(os.path.join(opt.result_dir,"pth"))
    
    print_options(parser, opt)
    # 初始化权重保存方法
    manager = CheckpointManager(save_dir=os.path.join(opt.result_dir,"pth"), max_keep=5)

    """ dataset and dataloader """
    train_dataset = LoadDatasets(img_path=opt.train_img_path,
                                 lab_path=opt.train_lab_path,
                                 dataset_name=opt.dataset_name,
                                 is_aug=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=4)
    val_dataset = LoadDatasets(img_path=opt.val_img_path,
                                 lab_path=opt.val_lab_path,
                                 dataset_name=opt.dataset_name,
                                 is_aug=False)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=4)
    print(f"Train dataset length:{len(train_loader)}\t,Val dataset length:{len(val_loader)}")

    """ device configuration """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    # select model name
    model = IVGNet(in_size=3,out_size=128,n_classes=3,
                        middle_size=128,device=device,
                        block_size_w=36,block_size_h=36)
    """set loss function"""
    Loss_CE = nn.CrossEntropyLoss()
    Loss_DSC = DiceLoss()
    Loss_FT = FocalTverskyLoss()
    Loss_DW = VE_Loss()

    print_network(model, opt)
    """model init or load checkpoint"""
    # model.load_state_dict(torch.load('xxxx.pth'))
    """ optimizer and scheduler """
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    # setting learing scheduler
    cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=opt.Epochs,
        eta_min=opt.weight_decay)
    """ training part """

    """如果指定多卡，则加载到多卡上"""
    if torch.cuda.device_count() > 1 and len(opt.CUDA_VISIBLE_DEVICES)>1:
        model = torch.nn.DataParallel(model)

    model = model.to(device)
    Loss_CE = Loss_CE.to(device)
    Loss_DSC = Loss_DSC.to(device)
    Loss_FT = Loss_FT.to(device)
    Loss_DW = Loss_DW.to(device)
    # best model pth
    best_valid_miou=0.0
    for e in range(opt.Epochs):
        with tqdm(total= len(train_loader)*opt.numbers) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(e + 1, opt.Epochs))
            model.train()
            loss_sum = 0
            for i in range(opt.numbers):
                for j, (img, lab, img_name) in enumerate(train_loader):
                    img = img.to(device)
                    lab = lab.to(device, dtype=torch.long)
                    if opt.model_name == "DEDCGCNEE":
                        out = model(img,img)
                    else:
                        out = model(img)
                    
                    if opt.loss_name == "CE_DSC":
                        full_loss = Loss_CE(out,lab)+0.6*Loss_DSC(out,lab)
                    elif opt.loss_name == "CE_DSC_FT":
                        full_loss = Loss_CE(out,lab)+0.6*Loss_DSC(out,lab)+0.3*Loss_FT(out,lab)
                    elif opt.loss_name == "CE_DSC_FT_VE":
                        full_loss = Loss_CE(out,lab)+0.6*Loss_DSC(out,lab)+0.3*Loss_FT(out,lab)+0.3*Loss_DW(out,lab)
                    optimizer.zero_grad()
                    full_loss.backward()
                    optimizer.step()
                    loss_sum += full_loss.item()
                    _tqdm.update(1)
                    # break
            l = loss_sum/(len(train_loader)*opt.numbers)
        cosineScheduler.step()
        with torch.no_grad():
            model.eval()
            val_miou_sum = 0
            pbar = tqdm(enumerate(BackgroundGenerator(val_loader)), total=len(val_loader))
            for j, (img, lab, img_name) in pbar:
                img = img.to(device)
                lab = lab.to(device, dtype=torch.long)
                lab = lab.squeeze(1).cpu().detach().numpy()

                out = model(img)

                pred = torch.argmax(out,dim=1)
                pred = pred.squeeze(1).cpu().detach().numpy()
                iou = cal_miou(pred,lab)
                val_miou_sum += iou
                # break

            # 存储一次测试
            # print(pred.shape)
            r = pred[0].copy()
            g = pred[0].copy()
            b = pred[0].copy()
            g[g!=0]=0
            r[r!=1]=0
            b[b!=2]=0
            b[b==2]=1
            result = cv.merge([r*255,g*255,b*255])
            cv.imwrite(os.path.join(opt.result_dir,"result",str(e)+".png"), np.uint8(result))
            
            val_miou = val_miou_sum / (len(val_loader))
            print("Step:{},\tloss:{}\tLR:{},\tValid_mIoU:{}".format(e,l,optimizer.state_dict()['param_groups'][0]['lr'],val_miou))
            # save best model
            torch.save(model.state_dict(),os.path.join(opt.result_dir,"pth",str(e)+'.pt'))
            manager.save(model, val_miou, e)
            print("save best model!")
