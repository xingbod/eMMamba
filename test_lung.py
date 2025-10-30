import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import cv2
import torch.nn as nn

from utils.dataloader import test_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "4,7"

import numpy as np
from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision

import logging
logger = logging.getLogger(__name__)

def mean_iou_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1)
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum - intersection

    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou


def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1)  # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)

    smooth = .001
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)
    return dice

def indicators(output, target):
    output_ = output > 0.5
    target_ = target > 0.5

    iou_ = jc(output_, target_)
    dice_ = dc(output_, target_)
    if not np.any(target_):
        hd_ = np.nan
        hd95_ = np.nan
    else:
        hd_ = hd(output_, target_)
        hd95_ = hd95(output_, target_)

    # hd_ = hd(output_, target_)
    # hd95_ = hd95(output_, target_)
    recall_ = recall(output_, target_)
    specificity_ = specificity(output_, target_)
    precision_ = precision(output_, target_)

    return iou_, dice_, hd_, hd95_, recall_, specificity_, precision_

################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default='mamba', help='model name')
    parser.add_argument('--testsize', type=int, default=384, help='testing size')
    opt = parser.parse_args()


    logging.basicConfig(filename='./test_logs/Lung/'+opt.model_name+'.log', level=logging.INFO)

   
    from lib.model.mamba import get_mamba
    model = get_mamba()
    model = model.cuda()
    model = nn.DataParallel(model)


    for epoch in range(3):
        pth_path = './model_pth/Lung/'+str(epoch)+'/' + opt.model_name + '/' + opt.model_name + '.pth'
        model.load_state_dict(torch.load(pth_path))
        model.eval()

        dice_bank = []
        iou_bank = []
        hd95_bank = []
        recall_bank = []
        specificity_bank = []
        precision_bank = []

        ##### put data_path here #####
        data_path = '/media/Storage2/zbw/data/covid19/Dataset/'+str(epoch)+'/test'
        
        ##### save_path #####
        save_path = './result_map/Lung/' + str(epoch) + '/' + opt.model_name+'/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        print('Evaluating ' + data_path)
        logger.info('Evaluating ' + data_path)

        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, opt.testsize)
        DSC = 0.0
        JACARD = 0.0
        for i in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
          
            [res1, res2, res3, res4] = model(image)[0]
            res = F.upsample(res1 + res2 + res3 + res4, size=gt.shape, mode='bilinear', align_corners=False)

            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path+name, res*255)

            iou_, dice_, hd_, hd95_, recall_, specificity_, precision_ = indicators(gt, res)

            dice_bank.append(dice_)
            iou_bank.append(iou_)
            hd95_bank.append(hd95_)
            recall_bank.append(recall_)
            specificity_bank.append(specificity_)
            precision_bank.append(precision_)
        print('-----------------------------number:  ' + str(epoch) +'-------------------------------------------')
        print('{}--dice: {:.4f}, iou: {:.4f}'.format(epoch, np.mean(dice_bank), np.mean(iou_bank)))
       
        logger.info('-----------------------------number:  ' + str(epoch) +'-------------------------------------------')
        logger.info('{}--dice: {:.4f}, iou: {:.4f}'.format(epoch, np.mean(dice_bank), np.mean(iou_bank)))
       
        logger.info('=======================================================================================')

