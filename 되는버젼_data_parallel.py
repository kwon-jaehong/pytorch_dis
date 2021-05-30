# https://github.com/MOONJOOYOUNG/pytorch_imagenet_multiprocessing-distributed

import torch
import scipy.io as sio
import numpy as np
import cv2
import copy
import argparse
import os
import time
import torch.optim as optim
from dataloader.dataset import ImageLoader_synthtext, collate
from utils import averager
from craft import CRAFT
from datetime import datetime

import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from multiprocessing import set_start_method

import util


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


parser = argparse.ArgumentParser(description='CRAFT HANGUL models')
parser.add_argument('--img_rootdir', default='/data/data/synthtext/SynthText/', type=str)
parser.add_argument('--gt_mat', default='/data/data/synthtext/SynthText/gt.mat', type=str)

parser.add_argument('--batch_size', default=80, type = int, help='')
parser.add_argument('--epoch', default=100, help='')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='') # 3.2768e-5



parser.add_argument('--store_sample', default='store', help='Where to store samples')
parser.add_argument('--displayInterval', type=int, default=20, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=2000, help='Interval to be displayed')


parser.add_argument('--workers', default=4, type=int, help='') # 총 일할 cpu
parser.add_argument("--gpu_devices", type=int, nargs='+', default=[0,1,2,3], help="")
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--rank', default=0, type=int, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')


parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')

args = parser.parse_args()



# GPU 설정
gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

best_acc1 = 0
def main():
    # DistributedDataParallel를 위한 분산 셋팅
    args = parser.parse_args()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    
    # set gpu
    args.gpu = gpu
    ngpus_per_node = torch.cuda.device_count()    
    print("Use GPU: {} for training".format(args.gpu))
    
    
    args.rank = args.rank * ngpus_per_node + gpu    
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    
    
    
    # making model
    net = CRAFT()
    torch.cuda.set_device(args.gpu)
    net.cuda(args.gpu)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
    
    ### 로스값 선언
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss(reduction='mean')
    
    
    print(args.gpu,"번째 데이터 로드 시작")
  
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    print(args.gpu,"번째 배치사이즈: ",args.batch_size," 일하는 woker : ",args.workers)
    
    dataset = ImageLoader_synthtext(args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    
    train_loader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,num_workers=args.workers,collate_fn=collate,pin_memory=True,sampler=train_sampler)
    print(args.gpu,"번째 데이터 로드 끝")
    


    
    
    ''' -------------------------logger 선언----------------------------'''
    train_logger = util.Logger('./train.log')
    valid_logger = util.Logger('./valid.log')
    train_time_logger = util.Logger('./train_time.log')
    valid_time_logger = util.Logger('./valid_time.log')
    
    
#     train(net, criterion, optimizer, train_sampler, train_loader, val_sampler, val_loader, args.gpu)

    
    
    
    ## 학습 시작
    
    net.train()
    args.start_epoch = 0
    for epoch in range(args.start_epoch, args.epoch):
        train_sampler.set_epoch(epoch)
        train(train_loader, net, criterion, optimizer, epoch, args, train_logger,train_time_logger)
        
#         is_best = acc1 > best_acc1
#         best_acc1 = max(acc1, best_acc1)
    

    
    
def train(train_loader, model, criterion, optimizer, epoch, args, logger, time_logger):
#     print(args.gpu,"번째 학습 시작")
    div=10
    ''' -------------------------averageMeter 선언.-----------------------------'''
    batch_time = util.AverageMeter('Time', ':6.3f')
    data_time = util.AverageMeter('Data', ':6.3f')
    losses = util.AverageMeter('Loss', ':.4f')

    ''' -------------------------출력 progress 선언.-----------------------------'''
    progress = util.ProgressMeter(len(train_loader),[batch_time, data_time, losses],prefix="Epoch: [{}]".format(epoch))
    
    
    ''' -------------------------학습 시작.-----------------------------'''
    
#     print(args.gpu,"번째 모델 트레인전")

    end = time.time()
    
    
#     print(args.gpu,"번째 모델 트레인 후")
    
    for i, data in enumerate(train_loader):
        
#         print(args.gpu,"번째 데이터 계산전")
        
        data_time.update(time.time() - end)
        
        
        img, char_label, interval_label = data
        
        
#         print(args.gpu,"번째 데이터 받아옴")
        
        img = img.cuda(args.gpu, non_blocking=True)
        char_label = char_label.cuda(args.gpu, non_blocking=True)
        interval_label = interval_label.cuda(args.gpu, non_blocking=True)
        
        
#         print(args.gpu,"번째 데이터 쿠다 넣음")
        
        img.requires_grad_()
        optimizer.zero_grad()
        
        
#         print(args.gpu,"번째 옵티 초기화")
        
        preds, _ = model(img)
        
        
#         print(args.gpu,"번째 데이터 계산후")
        
        
        cost_char = criterion(preds[:,:,:,0], char_label).sum()/div
        cost_interval = criterion(preds[:,:,:,1], interval_label).sum()/div
        cost = cost_char + cost_interval
        cost.backward()
        optimizer.step()

        average_gradients(model)
        
        
        
        reduced_loss = reduce_tensor(cost.data)
        losses.update(reduced_loss.item(), img.size(0))
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if dist.get_rank() == 0:
            progress.display(i)
                
    if dist.get_rank() == 0:
        logger.write([epoch, losses.avg])
        time_logger.write([epoch, batch_time.avg, data_time.avg])
        
def average_gradients(model):
    gpu_count=4
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= gpu_count
        
def reduce_tensor(tensor):
    gpu_count=4
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    # gpu 갯수로 나눠줌.
    rt /= gpu_count
    return rt

    
if __name__ == '__main__':
    main()
