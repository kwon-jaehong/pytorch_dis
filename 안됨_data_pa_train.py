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


from parallel import DataParallelCriterion


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# print("사용가능 GPU : "+str(torch.cuda.device_count()))
# set_start_method('spawn', force=True)
# 

parser = argparse.ArgumentParser(description='SynthText')
parser.add_argument('--img_rootdir', default='/data/data/synthtext/SynthText/', type=str)
parser.add_argument('--gt_mat', default='/data/data/synthtext/SynthText/gt.mat', type=str)
parser.add_argument('--go_on', default='', type=str)
parser.add_argument('--pre_model', default='', type=str)
parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
parser.add_argument('--store_sample', default='store', help='Where to store samples')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for critic')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--displayInterval', type=int, default=20, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=2000, help='Interval to be displayed')



parser.add_argument('--gpu', default='1,2,3,4', help='')
parser.add_argument('--world_size', type=int, default=1, help='')
parser.add_argument('--rank', type=int, default=0, help='')



args = parser.parse_args()

if not os.path.isdir(args.store_sample):
    os.system('mkdir {0}'.format(args.store_sample))

def main():
    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node,args=(ngpus_per_node, args))

def main_worker(gpu, ngpus_per_node, args):  
    print("잘됨")
    

    
    args.gpu = gpu
    torch.cuda.set_device(args.gpu)
#     print(args.rank * ngpus_per_node + gpu)
    print("Use GPU: {} for training".format(args.gpu))
    args.rank = args.rank * ngpus_per_node + gpu

    print(str(args.rank)+" 랭크 번호, "+str(args.world_size)+"월드 사이즈")
    
    print("디스트 초기화 전")

    dist.init_process_group(backend='nccl',init_method='tcp://127.0.0.1:48481',world_size=args.world_size,rank=args.rank)

    print("디스트 초기화 후")
    
    ## 찍어보기 위한 용도
    eval_saveimg_target_dir = "./es_img/"
    target_img_path = "./picture/pic1.jpg"
    target_img = cv2.imread(target_img_path)
    target_img = cv2.resize(target_img, (2240, 1260))
    target_img = torch.FloatTensor(target_img).cuda(args.gpu).permute(2, 0, 1).unsqueeze(0)

    print("데이터 로드 시작")

    dataset = ImageLoader_synthtext(args)
    
    dataset_sampler = DistributedSampler(dataset,rank=args.rank, num_replicas=world_size,)
    
    data_loader = torch.utils.data.DataLoader(dataset, args.batch_size, num_workers=0, collate_fn=collate,sampler=dataset_sampler)

    print("데이터 로드 끝")

    
    criterion = torch.nn.MSELoss(reduction='mean')
    criterion = DataParallelCriterion(criterion)
    criterion = criterion.cuda(args.gpu)
    
    
    craft = CRAFT(pretrained=True)

    # craft=craft.cuda()

    print("모델 생성 끝")

    if args.go_on != '':
        print('loading pretrained model from %s' % args.pre_model)
        craft.load_state_dict(torch.load(args.pre_model), strict=False)

    craft = craft.cuda(args.gpu)
    craft = DistributedDataParallel(craft, device_ids=[args.gpu])

    loss_avg = averager()
    optimizer = optim.Adam(craft.parameters(), lr=args.lr)

    print("옵티마이저 선언 끝")


    
    
    
    
    
    div = 10
    count = 0
    
    for epoch in range(args.epoch):
        print(epoch,"시작")
        train_iter = iter(data_loader)
        i = 0
        
        while i < len(data_loader):
            time0 = time.time()
            data = train_iter.next()
            
            craft.train()
            img, char_label, interval_label = data
            img = img.cuda(args.gpu)
            char_label = char_label.cuda(args.gpu)
            interval_label = interval_label.cuda(args.gpu)
            img.requires_grad_()
            optimizer.zero_grad()
            preds, _ = craft(img)
            cost_char = criterion(preds[:,:,:,0], char_label).sum()/div
            cost_interval = criterion(preds[:,:,:,1], interval_label).sum()/div
            cost = cost_char + cost_interval
            cost.backward()
            optimizer.step()
            
            loss_avg.add(cost)
            i += 1

            # do checkpointing
            if i % args.saveInterval == 0:
                torch.save(craft.state_dict(), '{0}/craft_{1}_{2}_{3}.pth'.format(args.store_sample, epoch, i, loss_avg.val()))

            if i % args.displayInterval == 0:
                
                craft.eval()
                output, _ = craft(target_img)
                char_label = output[:,:,:,0].squeeze()
                char_label = char_label.cpu().detach().numpy()
                char_label = np.clip(char_label, 0, 255).astype(np.uint8)
                char_label = cv2.applyColorMap(char_label, cv2.COLORMAP_JET)
                cv2.imwrite(eval_saveimg_target_dir+str(count)+'.jpg', char_label)
                count = count+1
                craft.train()
                
                
                print('[%d/%d][%d/%d] lr: %.4f Loss: %f Time: %f s' %
                    (epoch, args.epoch, i, len(data_loader), optimizer.param_groups[0]['lr'], loss_avg.val(), time.time()-time0))
                loss_avg.reset()

if __name__ == '__main__':
	main()