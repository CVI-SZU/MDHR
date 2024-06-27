import argparse
import torch.backends.cudnn as cudnn
import os
import argparse
import numpy as np
import torch
import random
import logging
import shutil

# ------------------------------

def print_conf(conf):
    """Print and save config
    It will print both current configs and default values(if different).
    It will save config into a text file / [checkpoints_dir] / config.txt
    """
    message = ''
    message += '------------------ config ----------------\n'
    for k, v in sorted(vars(conf).items()):
        comment = ''
        # default = self.parser.get_default(k)
        # if v != default:
        #     comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '------------------- End ---------------------'
    return message

def set_env(conf):
    # set seeding
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    torch.cuda.manual_seed(conf.seed)
    torch.cuda.manual_seed_all(conf.seed)
    if 'cudnn' in conf:
        torch.backends.cudnn.benchmark = cfg.cudnn
    else:
        torch.backends.cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.gpu_ids

def set_outdir(conf):
    default_outdir = os.path.join('result', conf.dataset)
    outdir = os.path.join(default_outdir, conf.exp_name, conf.backbone)
    prefix = '_fold_'+str(conf.fold) 
    outdir = os.path.join(outdir, prefix)
    ensure_dir(outdir)
    conf.outdir = outdir
    return conf

def ensure_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print('{} is created'.format(dir_name))

def set_logger(conf):
    #Set the logger to log info in terminal and file `log_path`.

    loglevel = logging.INFO

    if conf.evaluate:
        outname = 'test.log'
    else:
        outname = 'train.log'

    outdir = conf.outdir
    log_path = os.path.join(outdir, outname)

    logger = logging.getLogger()
    logger.setLevel(loglevel)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

    logging.info(print_conf(conf))
    logging.info('writting logs to file {}'.format(log_path))