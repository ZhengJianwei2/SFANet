import torch
from torch.autograd import Variable
from datetime import datetime
import os, argparse
from model.SFANet import SFANet
from data import get_loader
from utils import clip_gradient, adjust_lr

import pytorch_iou

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=60, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
opt = parser.parse_args()

print('Learning Rate: {}'.format(opt.lr))

model =SFANet()

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
#
image_root = './dataset/train_dataset/ORSI4199/image/'
gt_root = './dataset/train_dataset/ORSI4199/mask/'
edge_root = './dataset/train_dataset/ORSI4199/edge/'
# image_root = './dataset/train_dataset/ORSSD/image/'
# gt_root = './dataset/train_dataset/ORSSD/mask/'
# edge_root = './dataset/train_dataset/ORSSD/edge/'

train_loader = get_loader(image_root, gt_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average = True)


def train(train_loader, model, optimizer, epoch):
    model.train()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts, edges = pack
        images = Variable(images)
        gts = Variable(gts)
        edges = Variable(edges)
        images = images.cuda()
        gts = gts.cuda()
        edges = edges.cuda()

        s1, s2, s3, s1_sig, s2_sig, s3_sig, edge1, edge2 = model(images)
        # bce+iou
        loss1 = CE(s1, gts) + IOU(s1_sig, gts) + CE(edge1, edges)
        loss2 = CE(s2, gts) + IOU(s2_sig, gts) + CE(edge2, edges)
        loss3 = CE(s3, gts) + IOU(s3_sig, gts)

        loss = loss1 + loss2 + loss3

        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if i % 80 == 0 or i == total_step:
            print(
                '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}'.
                format(datetime.now(), epoch, opt.epoch, i, total_step,
                       opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss.data, loss1.data,
                       loss2.data))

    save_path = 'models/SFANet/ORSI4199/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # if (epoch+1) % 5 == 0:
    if (epoch+1) >= 40:
        torch.save(model.state_dict(), save_path + 'SFANet.pth' + '.%d' % epoch)

print("Let's go!")
for epoch in range(1, opt.epoch):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)
