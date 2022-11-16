import os, logging,sys
import time
import torch
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from common.utils import AverageMeter
from common.distributed import is_master
from common.distributed import synchronize
logger = logging.getLogger(__name__)


dic={}
cnt=0
def train(train_loader, model, criterion, optimizer, epoch,postprocess,args,lr_scheduler=None,best_mAP=0,writer=None,global_step=1):
    logger.info('training')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss = AverageMeter()
    
    model.train()

    end = time.time()

    
    for i,  (source_frame, target,angle) in enumerate(train_loader):
        
        
        # measure data loading time
        data_time.update(time.time() - end)
        source_frame=source_frame.cuda()
            
            
        label = target[4]
        label=torch.LongTensor(label)
        label= label.cuda()
        
        if args.head_query:
            angle=angle.cuda()
        
        if args.head_query:
            output=model(source_frame,query=angle)
        else:
            output = model(source_frame)

        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
         
        avg_loss.update(loss.item())

        postprocess.update(output.detach().cpu(), target)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % 100 == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=avg_loss))

    synchronize()
    postprocess.save()
    synchronize()
    mAP=0
    if is_master():
        mAP = postprocess.get_mAP()
                    
    return avg_loss.avg,mAP,best_mAP,global_step


                
    
def validate(val_loader,model, postprocess,args,mode='val',weight=None):
    logger.info('evaluating')
    batch_time = AverageMeter()
    avg_loss = AverageMeter()
    model.eval()
    end = time.time()
    
    for i, (source_frame, target,angle) in enumerate(val_loader):

        source_frame=source_frame.cuda()
        if not mode=='test':
            targets=target[4].cuda()
        with torch.no_grad():
            if args.head_query:
                angle=angle.cuda()
                output=model(source_frame,query=angle)
            else:
                output=model(source_frame)
                
            if not mode=='test':
                loss=F.cross_entropy(output,targets,weight=weight)
                    
        postprocess.update(output.detach().cpu(), target)
        if not mode=='test':
            avg_loss.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % 100 == 0:
            logger.info('Processed: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.avg:.3f}\t'.format(
                        i, len(val_loader), batch_time=batch_time,loss=avg_loss))
                 
    synchronize()
    postprocess.save()
    synchronize()
    
    if mode == 'val':
        mAP = None
        if is_master():
            mAP = postprocess.get_mAP()
        return mAP,avg_loss.avg

    if mode == 'test':
        print('generate pred.csv')
