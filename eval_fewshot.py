from __future__ import print_function

import os
import argparse
# import socket
import time
import sys

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models import model_dict, model_pool
from models.util import create_model, count_params

from dataset.mini_imagenet import ImageNet, MetaImageNet
from dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from dataset.cifar import CIFAR100, MetaCIFAR100
from dataset.transform_cfg import transforms_options, transforms_list

from eval.meta_eval import meta_test
from ptflops import get_model_complexity_info

def parse_option():

    # hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    # load pretrained model
    parser.add_argument('--model', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--model_path', type=str, default=None, help='absolute path to .pth model')

    # dataset
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet',
                                                                                'CIFAR-FS', 'FC100'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--data_root', type=str, default='data', metavar='N',
                        help='Root dataset')
    parser.add_argument('--num_workers', type=int, default=3, metavar='N',
                        help='Number of workers for dataloader')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')
    # specify architectures for DARTS space
    parser.add_argument('--layers', type=int, default=2, help='number of layers')
    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--genotype', type=str, default='', help='Cell genotype')

    # Parameters for Logistic regression
    parser.add_argument('--C', type=float, default=1.0, help='coefficient of Logistic Regression')
    parser.add_argument('--nonorm', action='store_false', dest='norm', help='if normalize feature, default: True')

    opt = parser.parse_args()

    if 'trainval' in opt.model_path:
        opt.use_trainval = True
    else:
        opt.use_trainval = False

    # set the path according to the environment
    # if hostname.startswith('visiongpu'):
    #     opt.data_root = '/data/vision/phillipi/rep-learn/{}'.format(opt.dataset)
    #     opt.data_aug = True
    # elif hostname.startswith('instance'):
    #     opt.data_root = '/mnt/globalssd/fewshot/{}'.format(opt.dataset)
    #     opt.data_aug = True
    # elif opt.data_root != 'data':
    #     opt.data_aug = True
    # else:
    #     raise NotImplementedError('server invalid: {}'.format(hostname))
    opt.data_root = '/home/yitew2/data/{}'.format(opt.dataset)
    opt.data_aug = True

    return opt


def main():

    opt = parse_option()

    # test loader
    args = opt
    args.batch_size = args.test_batch_size
    # args.n_aug_support_samples = 1

    if opt.dataset == 'miniImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans,
                                                  fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaImageNet(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans,
                                                 fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64
    elif opt.dataset == 'tieredImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        meta_testloader = DataLoader(MetaTieredImageNet(args=opt, partition='test',
                                                        train_transform=train_trans,
                                                        test_transform=test_trans,
                                                        fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaTieredImageNet(args=opt, partition='val',
                                                       train_transform=train_trans,
                                                       test_transform=test_trans,
                                                       fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 448
        else:
            n_cls = 351
    elif opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        train_trans, test_trans = transforms_options['D']
        meta_testloader = DataLoader(MetaCIFAR100(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans,
                                                  fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCIFAR100(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans,
                                                 fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            if opt.dataset == 'CIFAR-FS':
                n_cls = 64
            elif opt.dataset == 'FC100':
                n_cls = 60
            else:
                raise NotImplementedError('dataset not supported: {}'.format(opt.dataset))
    else:
        raise NotImplementedError(opt.dataset)

    support_xs, _, _, _ = next(iter(meta_testloader))
    batch_size, _, channel, height, width = support_xs.size()

    # Get input channel/size for creating augmentcnn model
    if opt.model == 'augmentcnn':
        assert height == width
        opt.n_input_channels = channel
        opt.input_size = height

    # load model
    model = create_model(opt.model, n_cls, opt.dataset, args=opt )
    ckpt = torch.load(opt.model_path)
    model.load_state_dict(ckpt['model'])

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    # Calculate model size & number of flops
    print('Number of parameters: {}'.format(count_params(model)))

    macs, params = get_model_complexity_info(model, (channel, height, width), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print('Use norm:{}'.format(args.norm))

    # evalation
    start = time.time()
    val_acc, val_std = meta_test(model, meta_valloader, is_norm=args.norm, C=opt.C)
    val_time = time.time() - start
    print('val_acc: {:.4f}, val_std: {:.4f}, time: {:.1f}'.format(val_acc, val_std,
                                                                  val_time))

    start = time.time()
    val_acc_feat, val_std_feat = meta_test(model, meta_valloader, use_logit=False, is_norm=args.norm, C=opt.C)
    val_time = time.time() - start
    print('val_acc_feat: {:.4f}, val_std: {:.4f}, time: {:.1f}'.format(val_acc_feat,
                                                                       val_std_feat,
                                                                       val_time))

    start = time.time()
    test_acc, test_std = meta_test(model, meta_testloader, is_norm=args.norm, C=opt.C)
    test_time = time.time() - start
    print('test_acc: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc, test_std,
                                                                    test_time))

    start = time.time()
    test_acc_feat, test_std_feat = meta_test(model, meta_testloader, use_logit=False, is_norm=args.norm, C=opt.C)
    test_time = time.time() - start
    print('test_acc_feat: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc_feat,
                                                                         test_std_feat,
                                                                         test_time))


if __name__ == '__main__':
    main()
