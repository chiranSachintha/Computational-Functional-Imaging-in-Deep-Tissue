
from __future__ import print_function
import argparse
import os
from math import log10
import numpy as np
import sys
import os
import random
from glob import glob
from skimage import io
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from networks import define_G, define_D, GANLoss, print_network
#from data import get_training_set, get_test_set
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as standard_transforms

#from model import UNet
#from torchvision.models import resnet18

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
args = {
    'num_class': 1,
    'ignore_label': 255,
    'num_gpus': 2,
    'start_epoch': 1,
    'num_epoch': 200,
    'batch_size': 6,
    'lr': 0.005,
    'lr_decay': 0.9,
    'dice': 0,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'snapshot': '',
    'opt': 'adam',
}

class depthwise_separable_conv(nn.Module):
    def __init__(self):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(30, 15, kernel_size=3, padding=1, groups=nin//2)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        return out


class Dataset(Dataset):
    def __init__(self, img_dir, transform=None):
        #self.img_anno_pairs = glob(img_dir)
        transforms_ = [standard_transforms.ToTensor()
                       #,standard_transforms.Normalize((26.704), (49.92))]
                       #standard_transforms.Normalize((26.704, 23.208), (49.92, 41.98))
                        ]

        self.transform = standard_transforms.Compose(transforms_)

        transforms_target  = [standard_transforms.ToTensor()
                       #,standard_transforms.Normalize((23.74), (45.17))
                           ]

        self.transform_target = standard_transforms.Compose(transforms_target)
        # self.transform_img = standard_transforms.Compose(
        #     standard_transforms.ToTensor(), standard_transforms.Normalize((26.704, 23.208), (49.92, 41.98)))
        # self.transform_target = standard_transforms.Compose(
        #     standard_transforms.ToTensor(), standard_transforms.Normalize((23.74), (45.17)))
        idx = 0
        file_img = open(img_dir, 'r')
        self.img_anno_pairs = {}
        for line in file_img:
            self.img_anno_pairs[idx] = line[0:-1]
            idx = idx + 1

    def __len__(self):
        return len(self.img_anno_pairs)

    def __getitem__(self, index):
        _img = np.zeros((240, 240, 30))
        for i in range(0,14)
            _img[:, :, i] = Image.open(self.img_anno_pairs[index]+'_'+i+'.png')
            _img[:, :, i+1] = Image.open(self.img_anno_pairs[index] + '_' + 'pattern'+i + '.png')
            #_img[:, :, 1] = Image.open(self.img_anno_pairs[index] + '_flair.png')
        _target = Image.open(self.img_anno_pairs[index] + '_gt.png')
        _img = self.transform(_img)
        _target = self.transform(_target)
        # _img = self.transform(np.array(_img)[None, :, :])
        # _target = self.transform(np.array(_target)[None, :, :])
        #print(np.unique(np.array(_target)))

        #_img = torch.from_numpy(np.array(_img)[None, :, :]).float()
        #_target = torch.from_numpy(np.array(_target)[None, :, :]).float()

        return _img, _target

# Training settings
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--dataset', default='facades-unet')
parser.add_argument('--batchSize', type=int, default=20, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--input_nc', type=int, default=2, help='input image channels')
parser.add_argument('--output_nc', type=int, default=1, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True, help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
# root_path = "dataset/"
# train_set = get_training_set(root_path + opt.dataset)
# test_set = get_test_set(root_path + opt.dataset)
# training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
# testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

img_dir = '/media/mmlab/data/mobarak/Brats/4C/train_4C_3C_XY.txt'
dataset_ = Dataset(img_dir=img_dir, transform=None)
training_data_loader = DataLoader(dataset=dataset_, batch_size=opt.batchSize, shuffle=True, num_workers=2)


gpu_ids = range(args['num_gpus'])
print('===> Building model')
netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, gpu_ids=[0,1]).cuda()
netD = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'batch', False, gpu_ids=[0,1]).cuda()


criterionGAN = GANLoss()
criterionL1 = nn.L1Loss()
criterionMSE = nn.MSELoss()

# setup optimizer
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

print('---------- Networks initialized -------------')
print_network(netG)
print_network(netD)
print('-----------------------------------------------')

real_a = torch.FloatTensor(opt.batchSize, opt.input_nc, 240, 240)
real_b = torch.FloatTensor(opt.batchSize, opt.output_nc, 240, 240)

if opt.cuda:
    netD = netD.cuda()
    netG = netG.cuda()
    #netG = torch.nn.parallel.DataParallel(netG, device_ids=gpu_ids)
    #netD = torch.nn.parallel.DataParallel(netD, device_ids=gpu_ids)
    criterionGAN = criterionGAN.cuda()
    criterionL1 = criterionL1.cuda()
    criterionMSE = criterionMSE.cuda()
    real_a = real_a.cuda()
    real_b = real_b.cuda()


real_a = Variable(real_a)
real_b = Variable(real_b)


def train(epoch):
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        real_a_cpu, real_b_cpu = batch[0], batch[1]
        #print(real_a_cpu.size(), real_b_cpu.size())
        real_a.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
        real_b.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)
        #print('sdhfsdhfkshdfd osdjfkjsdhfksdhfsd sdhfkjsdhfkjsdhfds fjh',real_a.size())
        fake_b = netG(real_a)
        #print(fake_b.size())
        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################

        optimizerD.zero_grad()
        
        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = netD.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = netD.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)
        
        # Combined loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()
       
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ##########################
        optimizerG.zero_grad()
        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = netD.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

         # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb
        
        loss_g = loss_g_gan + loss_g_l1
        
        loss_g.backward()

        optimizerG.step()

        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
            epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item()))


def test():
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0], volatile=True), Variable(batch[1], volatile=True)
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        prediction = netG(input)
        mse = criterionMSE(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


def checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
        os.mkdir(os.path.join("checkpoint", opt.dataset))
    net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth.tar".format(opt.dataset, epoch)
    net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth.tar".format(opt.dataset, epoch)
    torch.save(netG, net_g_model_out_path)
    torch.save(netD, net_d_model_out_path)
    print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))

for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    checkpoint(epoch)
    #test()
    # if epoch % 50 == 0:
    #     checkpoint(epoch)
