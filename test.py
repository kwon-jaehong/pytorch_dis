
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel



import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--gpu', type=int, default=4, help='Interval to be displayed')
# parser.add_argument('--gpu', type=int, default=2000, help='Interval to be displayed')
# args = parser.parse_args()



def main():
    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = 4
    args.rank = 0
    
    print("메인")
    mp.spawn(main_worker, nprocs=ngpus_per_node, 
             args=(ngpus_per_node, args))
    
    
def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    torch.cuda.set_device(args.gpu)
    
    print("Use GPU: {} for training".format(args.gpu))
    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend='nccl', 
                            init_method='tcp://127.0.0.1:80',
                            world_size=args.world_size, 
                            rank=args.rank)
    
#     model = Bert()
#     model.cuda(args.gpu)
#     model = DistributedDataParallel(model, device_ids=[args.gpu])

    acc = 0
    print("잘됨")
#     for i in range(args.num_epochs):
#         model = train(model)
#         acc = test(model, acc)
if __name__ == '__main__':
	main()