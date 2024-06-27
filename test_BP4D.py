import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from config.conf import set_logger, set_outdir, set_env
from model.model_BP4D import MDHR
from utils.utils import *
from dataset.dataset import video_BP4D_val
import argparse

def get_dataloader(conf):
    print('==> Preparing data...')

    valset = video_BP4D_val(root_path=conf.dataset_path, length=conf.length, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), crop_size=conf.crop_size)
    val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    return val_loader, len(valset)

# Val
def val(net, val_loader):
    net.eval()
    statistics_list = None
    for batch_idx, (inputs, labels, _) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            labels = labels.float()
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            _, _, _, _, outputs = net(inputs)

            update_list = statistics(outputs, labels.view(-1, conf.au_num).detach(), 0.5)
            statistics_list = update_statistics_list(statistics_list, update_list)

    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)

    return mean_f1_score, f1_score_list, mean_acc, acc_list

def main(conf):
    dataset_info = BP4D_infolist
    val_loader, val_data_num = get_dataloader(conf)

    net = MDHR(dataset=conf.dataset, num_classes=conf.au_num, backbone=conf.backbone, k=conf.k)

    if conf.resume != '':
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()

    val_mean_f1_score, val_f1_score, val_mean_acc, val_acc = val(net, val_loader)

    infostr = {
        'val_mean_f1_score {:.2f},val_mean_acc {:.2f} '.format(
            100. * val_mean_f1_score, 100. * val_mean_acc)}
    logging.info(infostr)
    infostr = {'F1-score-list:'}
    logging.info(infostr)
    infostr = dataset_info(val_f1_score)
    logging.info(infostr)
    infostr = {'Acc-list:'}
    logging.info(infostr)
    infostr = dataset_info(val_acc)
    logging.info(infostr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Testing')

    # Device and Seed
    parser.add_argument('--gpu_ids', default='4,7', type=str, help='GPU ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--seed', default=0, type=int, help='seeding for all random operation')

    # Datasets
    parser.add_argument('--dataset', default="BP4D", type=str, help="experiment dataset: BP4D / DISFA")
    parser.add_argument('--dataset_path', default='/path/to/BP4D_dataset/', type=str, help="root path to dataset")
    parser.add_argument('--fold', default=1, type=int, help="fold number, 1,2,3")

    # Network 
    parser.add_argument('--backbone', default='resnet', type=str, help= "backbone architecture: resnet / swin_transformer")
    parser.add_argument('--au_num', default=12, type=int, help='number of AUs')
    parser.add_argument('--k', default=5, type=int, help='number of adjacent frames')

    # Testing Param
    parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--num_workers', default=8, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--crop-size', default=224, type=int, help="crop size of train/test image data")
    parser.add_argument('--length',default=16, type=int, help='frame number of each clip')

    # Experiment
    parser.add_argument('--exp-name', default='test', type=str, help="experiment name for saving checkpoints")
    parser.add_argument('--resume', default='/path/to/best_model_fold1.pth', type=str, metavar='path', help='path to latest checkpoint')
    parser.add_argument('--evaluate', action='store_true', help='evaluate mode')

    conf = parser.parse_args()

    # Build environment
    set_env(conf)
    set_outdir(conf)
    set_logger(conf)

    main(conf)
