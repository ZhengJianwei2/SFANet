import torch
import os, argparse

import time
import numpy as np
import torch.nn.functional as F
from model.SFANet import SFANet
from data import test_dataset
from tqdm import trange
import imageio
from skimage import img_as_ubyte

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
opt = parser.parse_args(args=[])

dataset_path = './dataset/test_dataset/'

model = SFANet()
model.load_state_dict(torch.load('./models/ORSI4199/qyqNet.pth.48'), False)

test_datasets = ['ORSI4199']

for dataset in test_datasets:
    save_path = './result/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/image/'
    gt_root = dataset_path + dataset + '/mask/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    time_sum = 0
    for i in trange(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        time_start = time.time()
        res, s2, s3, s1_sig, s2_sig, s3_sig, edge1, edge2 = model(image)
        time_end = time.time()
        time_sum = time_sum+(time_end-time_start)

        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imsave(save_path+name, img_as_ubyte(res))
        if i == test_loader.size-1:
            print('Running time {:.5f}'.format(time_sum/test_loader.size))
            print('Average speed: {:.4f} fps'.format(test_loader.size/time_sum))