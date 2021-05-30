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


import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from multiprocessing import set_start_method


# gpu_devices='0,1,2,3'
# gpu_devices = ','.join([str(id) for id in gpu_devices])
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

## 월드 사이즈는 1 랭크는 0
# workers =4
# epochs = 90
# batch-size = 8
# world-size = 1
# rank = 0
# dist-backend = "ncll"
# gpu_count = 4
# url = 'tcp://127.0.0.1:2222'
# distributed = True
# ngpus_per_node = 4

# torch.cuda.empty_cache()


parser = argparse.ArgumentParser(description='SynthText')
parser.add_argument('--img_rootdir', default='/data/data/synthtext/SynthText/', type=str)
parser.add_argument('--gt_mat', default='/data/data/synthtext/SynthText/gt.mat', type=str)
parser.add_argument('--batch_size', type=int, default=48, help='input batch size')
parser.add_argument('--store_sample', default='store', help='Where to store samples')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for critic')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--displayInterval', type=int, default=20, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=2000, help='Interval to be displayed')




parser.add_argument('--gpu', default='0,1,2,3', help='')
parser.add_argument('--gpu_count', type=int, default=4, help='')
parser.add_argument('--world_size', type=int, default=1, help='')
parser.add_argument('--rank', type=int, default=0, help='')
parser.add_argument('--workers', type=int, default=16, help='')


args = parser.parse_args()




best_acc1 = 0
def main():
    ngpus_per_node = args.gpu_count
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp 

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    
    args.gpu = gpu
    

    args.rank = args.rank * ngpus_per_node + gpu
    
    print("Use GPU: {} for training, rank : {}".format(args.gpu,args.rank))
    
    
    ## 워커초기화
    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:2222",world_size=args.world_size, rank=args.rank)
    
    craft = CRAFT()
    criterion = torch.nn.MSELoss(reduction='mean').cuda(args.gpu)
    optimizer = optim.Adam(craft.parameters(), lr=args.lr)

    
#     torch.cuda.set_device(args.gpu)
    
    craft = craft.cuda()
    craft = torch.nn.parallel.DistributedDataParallel(craft)
#     craft = torch.nn.parallel.DistributedDataParallel(craft, device_ids=[args.gpu])
#     num_params = sum(p.numel() for p in craft.parameters() if p.requires_grad)
    ## 모델 선언 끝
    
    
    ## 데이터 셋 선언
    print(args.gpu,"번째 데이터 로드 시작")
  
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    print(args.gpu,"번째 배치사이즈: ",args.batch_size," 일하는 woker : ",args.workers)
    
    dataset = ImageLoader_synthtext(args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    
    train_loader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,num_workers=args.workers, pin_memory=True,sampler=train_sampler)
    print(args.gpu,"번째 데이터 로드 끝")
    
    ## 데이터셋 선언 끝
    
    
    ''' -------------------------logger 선언----------------------------'''
    train_logger = util.Logger(os.path.join(save_path, './train.log'))
    valid_logger = util.Logger(os.path.join(save_path, './valid.log'))
    train_time_logger = util.Logger(os.path.join(save_path, './train_time.log'))
    valid_time_logger = util.Logger(os.path.join(save_path, './valid_time.log'))
    
    
    
    ## 학습 시작
    args.start_epoch = 0
    for epoch in range(args.start_epoch, args.epoch):
        train_sampler.set_epoch(epoch)
        train(train_loader, model, criterion, optimizer, epoch, args, train_logger,train_time_logger)
        
#         is_best = acc1 > best_acc1
#         best_acc1 = max(acc1, best_acc1)
    
    
    
def train(train_loader, model, criterion, optimizer, epoch, args, logger, time_logger):
    ''' -------------------------averageMeter 선언.-----------------------------'''
    batch_time = util.AverageMeter('Time', ':6.3f')
    data_time = util.AverageMeter('Data', ':6.3f')
    losses = util.AverageMeter('Loss', ':.4f')

    ''' -------------------------출력 progress 선언.-----------------------------'''
    progress = util.ProgressMeter(len(train_loader),[batch_time, data_time, losses],prefix="Epoch: [{}]".format(epoch))
    
    
    ''' -------------------------학습 시작.-----------------------------'''
    model.train()
    end = time.time()
    
    for i, data in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        
        img, char_label, interval_label = data
        
        img = img.cuda(args.gpu, non_blocking=True)
        char_label = char_label.cuda(args.gpu, non_blocking=True)
        interval_label = interval_label.cuda(args.gpu, non_blocking=True)
        
        
        img.requires_grad_()
        optimizer.zero_grad()
        preds, _ = craft(img)
        cost_char = criterion(preds[:,:,:,0], char_label).sum()/div
        cost_interval = criterion(preds[:,:,:,1], interval_label).sum()/div
        cost = cost_char + cost_interval
        cost.backward()
        optimizer.step()
        
        average_gradients(craft)
        
        
        if dist.get_rank() == 0:
            if i % args.print_freq == 0:
                progress.display(i)
                
    if dist.get_rank() == 0:
        logger.write([epoch, losses.avg])
        time_logger.write([epoch, batch_time.avg, data_time.avg])
        
def average_gradients(model):
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= args.gpu_count
        
def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    # gpu 갯수로 나눠줌.
    rt /= args.gpu_count
    return rt


if __name__ == '__main__':
    main()
    