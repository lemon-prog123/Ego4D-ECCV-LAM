import os, torch, glob, subprocess, shutil
import torch.nn as nn
from torch import Tensor
from PIL.Image import merge
import pandas as pd
from torchvision import transforms
import torch.nn.functional as F
from common.metrics import run_evaluation
from common.distributed import is_master, synchronize
from math import cos, pi
from albumentations import (Blur,Flip,ShiftScaleRotate,GridDistortion,ElasticTransform,HorizontalFlip,CenterCrop,
                            HueSaturationValue,Transpose,RandomBrightnessContrast,CLAHE,RandomCrop,Cutout,CoarseDropout,Normalize,ToFloat,OneOf,Compose,Resize,RandomRain,RandomFog,Lambda
                            ,ChannelDropout,ISONoise,VerticalFlip,RandomGamma,RandomRotate90, RandomBrightness,RandomContrast,GaussianBlur,MotionBlur,RandomScale,RandomCropFromBorders,CenterCrop,ColorJitter,Rotate)
import torch.nn.functional as F
from typing import Callable, Optional

'''def adjust_learning_rate(optimizer, epoch, args):
    
    if args.transformer:
        decay = args.lr_drop_ratio if epoch%20==19 else 1.0
    elif args.mag:
        decay=args.lr_drop_ratio if epoch%10==9 else 1.0
    else:
        decay=args.lr_drop_ratio if epoch%20==19 else 1.0
    
    lr = args.lr * decay
    global current_lr
    current_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    args.lr = current_lr
    return current_lr'''

def adjust_learning_rate(optimizer, current_epoch, max_epoch, lr_min=1e-6, lr_max=1e-4, warmup=True):
    warmup_epoch =max_epoch/10 if warmup else 0
    
    if current_epoch <=warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    elif current_epoch < max_epoch:
        lr = lr_min + (lr_max - lr_min) * (
                    1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    else:
        lr = lr_min + (lr_max - lr_min) * (
                1 + cos(pi * (current_epoch-max_epoch) / (max_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_transform(is_train,pure=False,aug=False):
    if pure==False and aug==False:
        if is_train:
            transform = transforms.Compose([transforms.ToTensor(),
                                #transforms.RandomHorizontalFlip(),
                                transforms.Normalize(
                                mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])])
        else:
            transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(
                                mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])])
    elif aug==True and is_train==True:
        
        func1=RandomScale((-0.2,0.2))
        func2=RandomCropFromBorders(p=0.5)
        func3=Resize(224,224)
        func4=HorizontalFlip(p=0.5)
        
        func5=CenterCrop(196,196)
        func6=ColorJitter(0.5,0.5,0.5,0,p=1)
        func7=Rotate(limit=15,p=1)
        func8=GaussianBlur(p=0.3)

        trans=Compose([func1,func2,func3,func4])
        trans2=Compose([func5,func6,func7,func8,func3])
        
        transform = transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])])
        print('Dual View')
        return [trans,trans2,transform]
    else:
        print('Pure mode')
        transform = transforms.Compose([transforms.ToTensor()])
        
    return transform

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, reduction='mean',ignore_index=None, epsilon:float=1e-2,weight=None):
        super().__init__()
        self.class_dim=1 
        self.epsilon = epsilon
        self.reduction = reduction
        self.log_softmax=nn.LogSoftmax(dim=self.class_dim)
        self.nll_loss=nn.NLLLoss(reduction="none",weight=weight)
        self.ignore_index=ignore_index
        
    def forward(self, preds, target):
        n_class = preds.size()[self.class_dim]
        if self.ignore_index is not None:

            mask = torch.where(target != self.ignore_index)
            if len(mask)==1:
                preds = preds[mask[0]]
            elif len(mask)==2:
                preds = preds[mask[0], :, mask[1]]
            elif len(mask)==3:
                preds = preds[mask[0], :, mask[1], mask[2]]
            elif len(mask)==4:
                preds = preds[mask[0], :, mask[1], mask[2], mask[4]]
            target = target[mask]  
        log_preds = self.log_softmax(preds)
        target_loss = self.nll_loss(log_preds, target)*(1-self.epsilon-self.epsilon/(n_class-1))
        
        untarget_loss = -log_preds.sum(dim=self.class_dim)*(self.epsilon/(n_class-1))
        loss=target_loss+untarget_loss
        return self.reduce_loss(loss,self.reduction)
    
    def reduce_loss(self,loss, reduction='mean'):
        return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss
    
def get_transform_data():
    albumentations_transform = Compose([
    MotionBlur(blur_limit=7, always_apply=False, p=0.5),
    Resize(256, 256), 
    RandomCrop(224, 224),
    RandomBrightness(limit=0.3, always_apply=False, p=0.5),
    RandomContrast(limit=0.3, always_apply=False, p=0.5),
    RandomGamma(gamma_limit=(60, 150), eps=None, always_apply=False, p=0.5),
    HorizontalFlip(),
    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5)
   ])
    
    return albumentations_transform
class PostProcessor():
    def __init__(self, args,mode='val'):
        if mode=='val':
            self.exp_path = args.exp_path
        elif mode=='gazeval':
            self.exp_path = args.exp_path+'gaze'
        elif mode=='headval':
            self.exp_path = args.exp_path+'head'
        elif mode=='gaze':
            self.exp_path = args.train_path+'gaze'
        elif mode=='head':
            self.exp_path = args.train_path+'head'
        else:
            self.exp_path = args.train_path
        self.save_path = f'{self.exp_path}/tmp'
        
        if not os.path.exists(self.exp_path) and is_master():
            os.mkdir(self.exp_path)
        
        if not os.path.exists(self.save_path) and is_master():
            os.mkdir(self.save_path)
            
        self.groundtruth = []
        self.prediction = []
        self.groundtruthfile = f'{self.save_path}/gt.csv.rank.{args.rank}'
        self.predctionfile = f'{self.save_path}/pred.csv.rank.{args.rank}'

    def update(self, outputs, targets):
        # postprocess outputs of one minibatch
        outputs = F.softmax(outputs, dim=-1)
        for idx, scores in enumerate(outputs):
            uid = targets[0][idx]
            trackid = targets[1][idx]
            frameid = targets[2][idx].item()
            x1 = targets[3][0][idx].item()
            y1 = targets[3][1][idx].item()
            x2 = targets[3][2][idx].item()
            y2 = targets[3][3][idx].item()
            label = targets[4][idx].item()
            self.groundtruth.append([uid, frameid, x1, y1, x2, y2, trackid, label])
            self.prediction.append([uid, frameid, x1, y1, x2, y2, trackid, 1, scores[1].item()])

    def save(self):
        if os.path.exists(self.groundtruthfile):
            os.remove(self.groundtruthfile)
        if os.path.exists(self.predctionfile):
            os.remove(self.predctionfile)
        gt_df = pd.DataFrame(self.groundtruth)
        gt_df.to_csv(self.groundtruthfile, index=False, header=None)
        pred_df = pd.DataFrame(self.prediction)
        pred_df.to_csv(self.predctionfile, index=False, header=None)
        synchronize()

    def get_mAP(self):
        # merge csv
        merge_path = f'{self.exp_path}/result'
        if not os.path.exists(merge_path):
            os.mkdir(merge_path)

        gt_file = f'{merge_path}/gt.csv'
        if os.path.exists(gt_file):
            os.remove(gt_file)
        gts = glob.glob(f'{self.save_path}/gt.csv.rank.*')
        cmd = 'cat {} > {}'.format(' '.join(gts), gt_file)
        subprocess.call(cmd, shell=True)
        pred_file = f'{merge_path}/pred.csv'
        if os.path.exists(pred_file):
            os.remove(pred_file)
        preds = glob.glob(f'{self.save_path}/pred.csv.rank.*')
        cmd = 'cat {} > {}'.format(' '.join(preds), pred_file)
        subprocess.call(cmd, shell=True)
        shutil.rmtree(self.save_path)
        return run_evaluation(gt_file, pred_file)


class TestPostProcessor():
    def __init__(self, args):
        self.exp_path = args.exp_path
        self.save_path = f'{self.exp_path}/tmp'
        if not os.path.exists(self.save_path) and is_master():
            os.mkdir(self.save_path)
        self.prediction = []
        self.predctionfile = f'{self.save_path}/pred.csv.rank.{args.rank}'

    def update(self, outputs, targets):
        # postprocess outputs of one minibatch
        outputs = F.softmax(outputs, dim=-1)
        for idx, scores in enumerate(outputs):
            uid = targets[0][idx]
            trackid = targets[1][idx]
            unique_id = targets[2][idx]
            self.prediction.append([uid, unique_id, trackid, 1, scores[1].item()])

    def save(self):
        if os.path.exists(self.predctionfile):
            os.remove(self.predctionfile)
        pred_df = pd.DataFrame(self.prediction)
        pred_df.to_csv(self.predctionfile, index=False, header=None)
        synchronize()

        #merge csv
        merge_path = f'{self.exp_path}/result'
        if not os.path.exists(merge_path) and is_master():
            os.mkdir(merge_path)
        pred_file = f'{merge_path}/pred.csv'
        preds = glob.glob(f'{self.save_path}/pred.csv.rank.*')
        cmd = 'cat {} > {}'.format(' '.join(preds), pred_file)
        subprocess.call(cmd, shell=True)
        shutil.rmtree(self.save_path)


def save_checkpoint(state, save_path, is_best=False, is_dist=False):
    save_path = f'{save_path}/checkpoint'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    epoch = state['epoch']
    filename = f'{save_path}/epoch_{epoch}.pth'
    torch.save(state, filename)

    if is_best:
        if os.path.exists(f'{save_path}/best.pth'):
            os.remove(f'{save_path}/best.pth')
        torch.save(state, f'{save_path}/best.pth')


def spherical2cartesial(x): 
    output = torch.zeros(x.size(0),3)
    output[:,2] = -torch.cos(x[:,1])*torch.cos(x[:,0])
    output[:,0] = torch.cos(x[:,1])*torch.sin(x[:,0])
    output[:,1] = torch.sin(x[:,1])
    return output


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count