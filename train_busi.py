import os
import numpy as np
import argparse
from datetime import datetime
import logging

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.loss import CrossEntropyLoss

import matplotlib.pyplot as plt

from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

import random
def seed_torch(seed=1020):
    print("seed: "+str(seed))
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def test(model, data_path, model_name = 'PVT-CASCADE'):
    image_root = '{}/test/images/'.format(data_path)
    gt_root = '{}/test/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, opt.img_size)
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        
        res1, res2, res3, res4 = model(image)[0] # forward
        res = F.upsample(res1 + res2 + res3 + res4, size=gt.shape, mode='bilinear', align_corners=False) # additive aggregation and upsampling
  

        res = res.sigmoid().data.cpu().numpy().squeeze() # apply sigmoid aggregation for binary segmentation
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        # eval Dice
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

    return DSC / num1, num1

def train(train_loader, model, optimizer, epoch, test_path, model_name = 'PVT-CASCADE'):
    model.train()
    global best
    loss_record = AvgMeter()
    for i, pack in enumerate(train_loader):
        optimizer.zero_grad()
      
        [d0, d1, d2, d3], cam_edge = model(images)
        loss_edge = loss_func(cam_edge, egs)
        loss0 = structure_loss(d0, gts)
        loss1 = structure_loss(d1, gts)
        loss2 = structure_loss(d2, gts)
        loss3 = structure_loss(d3, gts)
        loss = loss_edge + loss0 + loss1 + loss2 + loss3
       

        # ---- backward ----
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        loss_record.update(loss.data, opt.batchsize)
                
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' loss: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record.show()))
    # save model 
    save_path = (train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # torch.save(model.state_dict(), save_path + '' + model_name + '-last.pth')
    # choose the best model

    global dict_plot
   
    if (epoch + 1) % 1 == 0:
        total_dice = 0
        total_images = 0

        dataset_dice, n_images = test(model, test_path, model_name)
        total_dice += (n_images*dataset_dice)
        total_images += n_images
        logging.info('epoch: {}, dice: {}'.format(epoch, dataset_dice))

        meandice = total_dice/total_images
        print('Validation dice score: {}'.format(meandice))
        logging.info('Validation dice score: {}'.format(meandice))
        if meandice > best:
            print('##################### Dice score improved from {} to {}'.format(best, meandice))
            logging.info('##################### Dice score improved from {} to {}'.format(best, meandice))
            best = meandice
            torch.save(model.state_dict(), save_path + '' + model_name + '.pth')
            # torch.save(model.state_dict(), save_path +str(epoch)+ '' + model_name + '-best.pth')
    
if __name__ == '__main__':

    # seed_torch()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default='mamba', help='model name')

    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=True, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')

    parser.add_argument('--img_size', type=int,
                        default=384, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=200, help='every n epochs decay learning rate')

    parser.add_argument('--data_path', type=str,
                        default='/media/Storage2/zbw/data/breast/Dataset/',
                        help='path to train dataset')


    opt = parser.parse_args()

    logging.basicConfig(filename='./logs/Breast/train_log_' + opt.model_name + '.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    print('batch_size:', str(opt.batchsize))
    for i in range(3):
        train_save = './model_pth/Breast/'+str(i)+'/'+opt.model_name+'/'

        # ---- build models ----
      
        from lib.model.mamba import get_mamba
        from utils.dataloader import get_obj_loader

        model = get_mamba()
        model = model.cuda()
        loss_func = torch.nn.BCEWithLogitsLoss()
      
        model = nn.DataParallel(model)
        best = 0
        params = model.parameters()
        if opt.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
        else:
            optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

        print(optimizer)
        image_root = '{}'.format(opt.data_path)+str(i)+'/train/images/'
        gt_root = '{}'.format(opt.data_path)+str(i)+'/train/masks/'

        if opt.model_name == 'CFA-Net' or (
                'mamba' in opt.model_name and opt.model_name not in ['mamba_base', 'mamba_a']):
            eg_root = '{}'.format(opt.data_path) + str(i) + '/train/edges/'

            train_loader = get_obj_loader(image_root, gt_root, eg_root, batchsize=opt.batchsize,
                                          trainsize=opt.img_size, num_workers=12)
        else:
            train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.img_size,
                                      augmentation=opt.augmentation)

        total_step = len(train_loader)

        print("#" * 20, "Start Training", "#" * 20)
        test_path = opt.data_path + str(i)
        for epoch in range(1, opt.epoch):
            adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
            train(train_loader, model, optimizer, epoch, test_path, model_name = opt.model_name)

    
