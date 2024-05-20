import os
import numpy as np
from torch.utils.data import DataLoader

import torch.optim as optim
from tqdm import tqdm
import logging
from config.conf import set_logger, set_outdir, set_env
from model.model import MDHR
from utils.utils import *
from dataset.dataset import video_BP4D_train, video_BP4D_val
import torch.nn.functional as F
import argparse

def get_dataloader(conf):
    print('==> Preparing data...')

    trainset = video_BP4D_train(root_path=conf.dataset_path, length=conf.length, fold=conf.fold, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size)
    train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)

    valset = video_BP4D_val(root_path=conf.dataset_path, length=conf.length, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), crop_size=conf.crop_size)
    val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    return train_loader, val_loader, len(trainset), len(valset)

# Train
def train(conf, net, train_loader, optimizer, epoch, criterion_CE, criterion_WA):
    losses_ce = AverageMeter()
    losses_wa = AverageMeter()
    net.train()
    train_loader_len = len(train_loader)
    for batch_idx, (inputs,  labels, sub_labels) in enumerate(tqdm(train_loader)):
        adjust_learning_rate(optimizer, epoch, conf.epochs, conf.learning_rate, batch_idx, train_loader_len, conf.warmup_epoch)
        labels = labels.float()
        sub_labels = sub_labels.view(-1,4).long()
        if torch.cuda.is_available():
            inputs, labels, sub_labels = inputs.cuda(), labels.cuda(), sub_labels.cuda()

        optimizer.zero_grad()

        up_pred, mid_pred, down1_pred, down2_pred, outputs = net(inputs)
        loss_ce = (criterion_CE(up_pred, sub_labels[:,0]) + criterion_CE(mid_pred, sub_labels[:,1]) + criterion_CE(down1_pred,sub_labels[:,2]) + criterion_CE(down2_pred,sub_labels[:,3]))/4.
        loss_wa = criterion_WA(outputs, labels.view(-1, conf.au_num))
        loss = conf.Lambda * loss_ce + loss_wa

        loss.backward()
        optimizer.step()

        losses_ce.update(loss_ce.data.item(), outputs.size(0))
        losses_wa.update(loss_wa.data.item(), outputs.size(0))
    return losses_ce.avg, losses_wa.avg

# Val
def val(net, val_loader, criterion_WA):
    losses_wa = AverageMeter()
    net.eval()
    statistics_list = None

    for batch_idx, (inputs, labels, sub_labels) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            labels = labels.float()
            sub_labels = sub_labels.view(-1, 4).long()
            if torch.cuda.is_available():
                inputs, labels, sub_labels = inputs.cuda(), labels.cuda(), sub_labels.cuda()

            up_pred, mid_pred, down1_pred, down2_pred, outputs = net(inputs)

            loss_wa = criterion_WA(outputs, labels.view(-1, conf.au_num))
            losses_wa.update(loss_wa.data.item(), outputs.size(0))

            update_list = statistics(outputs, labels.view(-1, conf.au_num).detach(), 0.5)
            statistics_list = update_statistics_list(statistics_list, update_list)

    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)
    return losses_wa.avg, mean_f1_score, f1_score_list, mean_acc, acc_list

def main(conf):

    dataset_info = BP4D_infolist

    start_epoch = 0
    train_loader, val_loader, train_data_num, val_data_num = get_dataloader(conf)
    train_weight = torch.from_numpy(np.loadtxt(os.path.join(conf.dataset_path, conf.dataset+'_train_weight_fold'+str(conf.fold)+'.txt')))

    net = MDHR(dataset=conf.dataset, num_classes=conf.au_num, backbone=conf.backbone, k=conf.k)

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()
        train_weight = train_weight.cuda()

    criterion_CE = nn.CrossEntropyLoss()
    criterion_WA = WeightedAsymmetricLoss(weight=train_weight)
    
    optimizer = optim.AdamW(net.parameters(), betas=(0.9, 0.999), lr=conf.learning_rate, weight_decay=conf.weight_decay)

    best_f1 = 0
    best_epoch = -1
    best_acc = 0

    for epoch in range(start_epoch, conf.epochs):
        lr = optimizer.param_groups[0]['lr']
        print("Epoch: [{} | {} LR: {} ]".format(epoch + 1, conf.epochs, lr))
        train_loss_ce, train_loss_wa = train(conf, net, train_loader, optimizer, epoch, criterion_CE, criterion_WA)
        infostr = {'Epoch:  {}   train_loss_CE: {:.5f}  train_loss_WA: {:.5f} '.format(epoch + 1, train_loss_ce, train_loss_wa)}
        logging.info(infostr)

        # val and save checkpoints
        if (epoch+1) % conf.val_interval == 0:
            val_loss, val_mean_f1_score, val_f1_score, val_mean_acc, val_acc = val(net, val_loader, criterion_WA)
            if val_mean_f1_score > best_f1:
                best_f1 = val_mean_f1_score
                best_acc = val_mean_acc
                best_epoch = epoch+1
                checkpoint = {
                     'epoch': epoch,
                     'state_dict': net.state_dict(),
                     'optimizer': optimizer.state_dict(),
                 }
                torch.save(checkpoint, os.path.join(conf.outdir, 'best' + '_model.pth'))
            infostr = {
                'epoch: {} val_loss: {:.5f} val_mean_f1_score {:.2f},val_mean_acc {:.2f} '.format(
                    epoch + 1, val_loss,  100. * val_mean_f1_score, 100. * val_mean_acc)}
            logging.info(infostr)
            infostr = {'F1-score-list:'}
            logging.info(infostr)
            infostr = dataset_info(val_f1_score)
            logging.info(infostr)
            infostr = {'Acc-list:'}
            logging.info(infostr)
            infostr = dataset_info(val_acc)
            logging.info(infostr)

    infostr = {'best F1-score :{:.5f} and acc:{:.5f} in epoch :{}'.format(best_f1, best_acc, best_epoch)}
    logging.info(infostr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')

    # Device and Seed
    parser.add_argument('--gpu_ids', default='1,2', type=str, help='GPU ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--seed', default=0, type=int, help='seeding for all random operation')

    # Datasets
    parser.add_argument('--dataset', default="BP4D", type=str, help="experiment dataset: BP4D / DISFA / Aff")
    parser.add_argument('--dataset_path', default='/data/wangzihan/Dataset/BP4D_dataset/', type=str, help="root path to dataset")
    parser.add_argument('--fold', default=1, type=int, help="fold number, 1,2,3")

    # Network 
    parser.add_argument('--backbone', default='resnet', type=str, help= "backbone architecture: resnet / swin_transformer")
    parser.add_argument('--au_num', default=12, type=int, help='number of AUs')
    parser.add_argument('--k', default=5, type=int, help='number of adjacent frames')

    # Training Param
    parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('-lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('-e', '--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--warmup_epoch', default=0, type=int, help='number of warm-up epochs')
    parser.add_argument('--val_interval', default=25, type=int, help='validation interval epochs')
    parser.add_argument('--num_workers', default=8, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--weight-decay', '-wd', default=5e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument('--optimizer-eps', default=1e-8, type=float, help='optimizer epsilon')
    parser.add_argument('--crop-size', default=224, type=int, help="crop size of train/test image data")
    parser.add_argument('--length',default=16, type=int, help='frame number of each clip')
    parser.add_argument('--Lambda', default=0.01, type=float, help='balanced weight of loss')

    # Experiment
    parser.add_argument('--exp-name', default='train', type=str, help="experiment name for saving checkpoints")
    parser.add_argument('--resume', default='', type=str, metavar='path', help='path to latest checkpoint')
    parser.add_argument('--evaluate', action='store_true', help='evaluate mode')

    conf = parser.parse_args()

    # Build environment
    set_env(conf)
    set_outdir(conf)
    set_logger(conf)

    main(conf)
