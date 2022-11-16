from ast import arg
import os, sys, random, pprint
import torch
import torch.optim
import torch.utils.data
from torch.utils.data import DistributedSampler
from dataset.data_loader import  ImagerLoader, TestImagerLoader
from model.model import GazeDETR
from common.config import argparser
from common.logger import create_logger
from common.engine import train, validate
from common.utils import PostProcessor, get_transform, save_checkpoint, TestPostProcessor
from common.distributed import distributed_init, is_master, synchronize
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter


def main(args):
    print('PID')
    print(os.getpid())
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
        torch.backends.cudnn.enabled = True
        torch.cuda.init()
    
    if args.dist:
        distributed_init(args)

    if is_master() and not os.path.exists(args.exp_path):
        os.mkdir(args.exp_path)
        
    # seed
    m_seed = 13
    # set seed
    torch.manual_seed(m_seed)
    torch.cuda.manual_seed_all(m_seed)
    
    logger = create_logger(args)
    logger.info(pprint.pformat(args))

    logger.info(f'Model: {args.model}')
    model = eval(args.model)(args)

    if args.dist:
        
        model= torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(args.device_id)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.rank],
            output_device=args.rank,
            find_unused_parameters=False)
    
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    if not args.eval:
        
        if not args.val and not args.pesudo:
            train_dataset = ImagerLoader(args.source_path, args.train_file, args.json_path,
                                     args.gt_path, stride=args.train_stride, transform=get_transform(True,pure=args.pure,aug=args.DR),args=args)
                
        val_dataset = ImagerLoader(args.source_path, args.val_file, args.json_path, args.gt_path,
                                   stride=args.val_stride, mode='val', transform=get_transform(False,pure=args.pure),args=args)
    
        params = {}
        if args.dist:
            params = {'sampler': DistributedSampler(train_dataset)}
        else:
            params = {'shuffle': True}
            
        if not args.val:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=False,
                **params)
            test_loader=None
                
        if args.dist:
            params = {'sampler': DistributedSampler(val_dataset, shuffle=False)}
        else:
            params = {'shuffle': False}

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=False,
            **params)

        print('PID')
        print(os.getpid())
        
    else:
        print('PID')
        print(os.getpid())
        test_dataset = TestImagerLoader(args.test_path,args,stride=args.test_stride, transform=get_transform(False,pure=args.pure))

        if args.dist:
            params = {'sampler': DistributedSampler(test_dataset, shuffle=False)}
        else:
            params = {'shuffle': False}

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            **params)
    if not args.val and not args.eval:
        lam=train_dataset.lam
        nlam=train_dataset.nlam
        nlam=lam/(lam+nlam)
        lam=1-nlam
        class_weights=torch.FloatTensor([nlam,lam]).cuda()
    else:
        class_weights = torch.FloatTensor(args.weights).cuda()
    
    print(class_weights)
    
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    if not args.dist:
        args.world_size=1
    
    

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr*args.world_size)
    
            
    best_mAP = 0
    if not args.val and not args.eval:
        CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=(5e-6)*args.world_size)
        
    
    synchronize()
    train_loss=[]
    val_loss=[]
    val_map=[]
    train_map=[]
    global_step=1
    if not args.eval and not args.val:
        logger.info('start training')
        writer = SummaryWriter(log_dir="runs/"+args.train_path, flush_secs=120)
        
        for epoch in range(args.epochs):
            
            for param_group in optimizer.param_groups:
                print("Now lr:")
                print(param_group['lr'])
            
            if args.dist:
                train_loader.sampler.set_epoch(epoch)
            # train for one epoch
            postprocess_train = PostProcessor(args,mode='train')
            
            loss,mAP,mAP_epoch,global_step=train(train_loader, model, criterion, optimizer, epoch,postprocess_train,
                                                       args,CosineLR,best_mAP=best_mAP,
                                                       writer=writer,global_step=global_step)
            best_mAP = max(best_mAP, mAP_epoch)
            
            writer.add_scalar(tag='Loss/train',
                          scalar_value=loss,
                           global_step=global_step)
            writer.add_scalar(tag='Lr/train',
                          scalar_value=optimizer.param_groups[0]['lr'],
                           global_step=global_step)
            writer.add_scalar(tag='mAP/train',
                          scalar_value=mAP*100,
                           global_step=epoch)
            
            train_loss.append(loss)
            train_map.append(mAP)
            
            
            if is_master():
                logger.info(f'mAP: {mAP:.4f}')
            
            # evaluate on validation set
            postprocess_val = PostProcessor(args)
                
                        
            mAP,loss = validate(val_loader,model, postprocess_val, args=args,mode='val',weight=class_weights)
            
            
            val_map.append(mAP)
            val_loss.append(loss)
            writer.add_scalar(tag='mAP/val',
                          scalar_value=mAP*100,
                           global_step=global_step)
            
            global_step+=1
            print('Global Step:'+str(global_step))

            print(train_loss)
            print(train_map)
            print(val_map)
            print(val_loss)
            
            if is_master():
                # remember best mAP in validation and save checkpoint
                is_best = mAP > best_mAP
                best_mAP = max(mAP, best_mAP)
                logger.info(f'mAP: {mAP:.4f} best mAP: {best_mAP:.4f}')

                save_checkpoint({
                    'epoch': global_step,
                    'state_dict': model.state_dict(),
                    'mAP': mAP},
                    save_path=args.exp_path,
                    is_best=is_best,
                    is_dist=args.dist)
                
            if not args.warmup and CosineLR!=None:
                CosineLR.step()
            synchronize()
    elif args.val:
        
        postprocess = PostProcessor(args)
        mAP,loss = validate(val_loader,model, postprocess, args=args,mode='val',weight=class_weights)
        val_map.append(mAP)
        val_loss.append(loss)
        
        if is_master():
                # remember best mAP in validation and save checkpoint
            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            if args.model=="BaselineLSTM":
                logger.info(f'mAP: {mAP:.4f} best mAP: {best_mAP:.4f}')
                logger.info(f'mean: {model.running_mean.item():.4f} var: {model.running_var.item():.4f}')
            
        synchronize()
        
    else:
        logger.info('start evaluating')
        postprocess = TestPostProcessor(args)
        validate(test_loader,model, postprocess, mode='test',args=args,weight=class_weights)
    
    print(train_loss)
    print(val_map)
    print(val_loss)


def distributed_main(device_id, args):
    args.rank = args.start_rank + device_id
    args.device_id = device_id
    main(args)


def run():
    args = argparser.parse_args()

    if args.dist:
        args.world_size = max(1, torch.cuda.device_count())
        assert args.world_size <= torch.cuda.device_count()

        if args.world_size > 0 and torch.cuda.device_count() > 1:
            port = random.randint(10000, 20000)
            args.init_method = f"tcp://localhost:{port}"
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args,),
                nprocs=args.world_size,
            )
    else:
        main(args)


if __name__ == '__main__':
    run()
