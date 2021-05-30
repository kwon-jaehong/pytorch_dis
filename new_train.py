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

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

print(torch.cuda.device_count())


parser = argparse.ArgumentParser(description='SynthText')
parser.add_argument('--img_rootdir', default='/data/data/synthtext/SynthText/', type=str)
parser.add_argument('--gt_mat', default='/data/data/synthtext/SynthText/gt.mat', type=str)
parser.add_argument('--go_on', default='', type=str)
parser.add_argument('--pre_model', default='', type=str)
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--store_sample', default='store', help='Where to store samples')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for critic')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--displayInterval', type=int, default=20, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=2000, help='Interval to be displayed')
args = parser.parse_args()

if not os.path.isdir(args.store_sample):
    os.system('mkdir {0}'.format(args.store_sample))

## 찍어보기 위한 용도
eval_saveimg_target_dir = "./es_img/"
target_img_path = "./picture/pic1.jpg"
target_img = cv2.imread(target_img_path)
target_img = cv2.resize(target_img, (2240, 1260))
target_img = torch.FloatTensor(target_img).cuda().permute(2, 0, 1).unsqueeze(0)
    
print("데이터 로드 시작")

dataset = ImageLoader_synthtext(args)
assert dataset
data_loader = torch.utils.data.DataLoader(dataset, args.batch_size, num_workers=0, shuffle=True, collate_fn=collate)

print("데이터 로드 끝")

criterion = torch.nn.MSELoss(reduction='mean')
criterion = criterion.cuda()
craft = CRAFT(pretrained=True)

# craft=craft.cuda()

print("모델 생성 끝")

if args.go_on != '':
    print('loading pretrained model from %s' % args.pre_model)
    craft.load_state_dict(torch.load(args.pre_model), strict=False)

craft = torch.nn.DataParallel(craft)
craft = craft.cuda()

loss_avg = averager()
optimizer = optim.Adam(craft.parameters(), lr=args.lr)

print("옵티마이저 선언 끝")

def train_batch(data):
    div = 10
    craft.train()
    img, char_label, interval_label = data
    img = img.cuda()
    char_label = char_label.cuda()
    interval_label = interval_label.cuda()

    img.requires_grad_()
    optimizer.zero_grad()
    preds, _ = craft(img)
    cost_char = criterion(preds[:,:,:,0], char_label).sum()/div
    cost_interval = criterion(preds[:,:,:,1], interval_label).sum()/div
    cost = cost_char + cost_interval
    cost.backward()
    optimizer.step()
    return cost

def main():    
    count = 0
    for epoch in range(args.epoch):
        print(epoch,"시작")
        train_iter = iter(data_loader)
        i = 0
        while i < len(data_loader):
            time0 = time.time()
            data = train_iter.next()
            cost = train_batch(data)
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
                cv2.imwrite(eval_saveimg_target_dir+str(count)+'_iter_'+str(i)+'.jpg', char_label)
                count = count+1
                craft.train()
                
                
                print('[%d/%d][%d/%d] lr: %.4f Loss: %f Time: %f s' %
                    (epoch, args.epoch, i, len(data_loader), optimizer.param_groups[0]['lr'], loss_avg.val(), time.time()-time0))
                loss_avg.reset()

if __name__ == '__main__':
	main()