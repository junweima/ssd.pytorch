from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss, MultiBoxKDLoss
from ssd import build_ssd
from ssd_mb import build_mobile_ssd
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
# parser.add_argument('--pretrained_vgg', default='ssd300_mAP_77.43_v2.pth',
                    help='Pretrained base model')
# parser.add_argument('--basenet', default='mobilenet_v2.pth.tar',
                    # help='Pretrained base model')
parser.add_argument('--batch_size', default=23, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=16, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--mbv2_base', default=False, type=str2bool, 
                    help='whether using mbv2 base or not')
parser.add_argument('--train_kd', default=False, type=str2bool,
                    help='training knowledge distillation')
parser.add_argument('--input_mixup', default=False, type=str2bool,
                    help='training with input mixup')
parser.add_argument('--manifold_mixup', default=False, type=str2bool,
                    help='training with manifold mixup')
parser.add_argument('--hint_learning', default=False, type=str2bool,
                    help='use hint based learning to pre-train student\'s feature map')
parser.add_argument('--hint_matching', default=False, type=str2bool,
                    help='use hint based learning to pre-train student\'s feature map')
parser.add_argument('--limited', default=True, type=str2bool,
                    help='limit input to 8000 (half)')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def apply_input_mixup(images, alpha=0.2):
    batch_size = images.shape[0]
    mixup_lambda = np.random.beta(alpha, alpha)
    batch1 = np.random.choice(batch_size, batch_size)
    batch2 = np.random.choice(batch_size, batch_size)
    images = mixup_lambda * images[batch1, ...] + (1-mixup_lambda) * images[batch2, ...]
    return images


def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        if args.mbv2_base:
            cfg = voc_mb_v2
        else:
            cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))

    if args.limited:
        dataset.ids = dataset.ids[:4000]

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    if not args.mbv2_base:
        ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    else:
        ssd_net = build_mobile_ssd('train', cfg['min_dim'], cfg['num_classes'])

    if args.train_kd:
        teacher = build_ssd('train', cfg['min_dim'], cfg['num_classes'])

    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    elif not args.mbv2_base: # load vgg weights
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)
    else: # load mbv2 weights
        mbv2_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.mbv2.load_state_dict(mbv2_weights)

    if args.train_kd:
        print('Using teacher pretrained weights, loading {}...'.format(args.pretrained_vgg))
        teacher.load_weights(args.save_folder + args.pretrained_vgg)

    if args.cuda:
        net = net.cuda()
        if args.train_kd:
            teacher = teacher.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)
    kl_criterion = nn.KLDivLoss(size_average=False)

    net.train()
    if args.train_kd:
        teacher.eval()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    if args.hint_learning:
        n_iter = cfg['pretrain_iter']
    else:
        n_iter = cfg['max_iter']

    print('\n niter is {} \n'.format(n_iter))
    for iteration in range(args.start_iter, n_iter):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        # images, targets = next(batch_iterator)
        try:
            images, targets = next(batch_iterator)
            if args.train_kd and args.input_mixup:
                images = apply_input_mixup(images)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)
            if args.train_kd and args.input_mixup:
                images = apply_input_mixup(images)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        
        # calculate mixup batches
        if args.manifold_mixup:
            batch_size = images.shape[0]
            mixup_batches = [
                np.random.choice(batch_size, batch_size),
                np.random.choice(batch_size, batch_size)
            ]
            alpha = 0.2
            mixup_lambda = np.random.beta(alpha, alpha)
        else:
            mixup_batches = None
            mixup_lambda = None

        # forward 
        t0 = time.time()
        if not args.mbv2_base:
            out, guided, _ = net(images, apply_mixup=args.manifold_mixup, mixup_batches=mixup_batches, mixup_lambda=mixup_lambda)
        else:
            out, guided = net(images, apply_mixup=args.manifold_mixup, mixup_batches=mixup_batches, mixup_lambda=mixup_lambda)

        # backprop
        optimizer.zero_grad()
        
        if args.train_kd and not args.hint_learning:
            t_out, detections, hint = teacher(images, apply_mixup=args.manifold_mixup, mixup_batches=mixup_batches, mixup_lambda=mixup_lambda)

            # import pdb; pdb.set_trace()

            t_targets = [] # the final teacher detection targets for all images
            for i_image in range(detections.size(0)): 
                t_target = None # teacher detection target for 1 image
                for i_class in range(1, detections.size(1)):
                    t_mask = (detections[i_image, i_class, :, 0] >= 0.6)
                    # import pdb; pdb.set_trace()
                    if t_target is None:
                        t_dets = detections[i_image, i_class, t_mask, 1:]
                        if t_dets.size(0) == 0:
                            continue
                        t_class = torch.Tensor([float(i_class-1)]).unsqueeze(0).expand_as(t_dets[:,0].unsqueeze(1))
                        t_target = torch.cat((t_dets, t_class), 1).contiguous()
                    else:
                        t_dets = detections[i_image, i_class, t_mask, 1:]
                        if t_dets.size(0) == 0:
                            continue
                        t_class = torch.Tensor([float(i_class-1)]).unsqueeze(0).expand_as(t_dets[:,0].unsqueeze(1))
                        add_target = torch.cat((t_dets, t_class), 1).contiguous()
                        t_target = torch.cat((t_target, add_target), 0).contiguous()
                # handle the case when all of the teacher's predictions are below conf threshold = 0.6
                if t_target is None:
                    method = 1
                    if method == 0:
                        idx = detections[i_image, 1:, :, 0].view(-1).sort(descending=True)
                        n_prior_boxes = detections.size(2)
                        t_dets = detections[i_image, int(idx[1][0]/n_prior_boxes), idx[1][0]%n_prior_boxes, 1:].unsqueeze(0)
                        t_class = (torch.Tensor([float(int(idx[1][0]/n_prior_boxes))])).unsqueeze(0)
                        if t_target is None:
                            t_target = torch.cat((t_dets, t_class), 1).contiguous()
                        else:
                            add_target = torch.cat((t_dets, t_class), 1).contiguous()
                            t_target = torch.cat((t_target, add_target), 0).contiguous()
                    elif method == 1:
                        t_target = targets[i_image]

                # import pdb; pdb.set_trace()
                t_targets.append(t_target.detach())
            # knowledge distillation loss
            loss_l, loss_c = criterion(out, t_targets)
        elif args.train_kd and args.hint_learning:
            _, fmap_t = teacher(images, apply_mixup=False, mixup_batches=None)
            out, _ = net(images, apply_mixup=False, mixup_batches=None)
            # TODO: LOSS
        else:
            loss_l, loss_c = criterion(out, targets)
        
        loss = loss_l + loss_c

        # hint matching adds additional loss for matching feature maps
        if args.train_kd and args.hint_matching:
            hint_loss = F.mse_loss(guided.view(-1), hint.view(-1).detach(), size_average=True)
            loss += hint_loss.double() * 0.1
            # import pdb; pdb.set_trace()

        # feature map loss = loc loss + conf loss
        if args.train_kd:
            num_prior_boxes = out[0].shape[1]
            fmap_loc_loss = F.smooth_l1_loss(out[0], t_out[0].detach(), size_average=False)
            fmap_conf_loss = kl_criterion(F.log_softmax(out[1].view(-1, cfg['num_classes']), 1), 
                                F.softmax(t_out[1].view(-1, cfg['num_classes']), 1).detach())
            loss += (fmap_loc_loss.double() + fmap_conf_loss.double())/num_prior_boxes

        # import pdb; pdb.set_trace()

        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')

        if args.visdom:
            update_vis_plot(iteration, loss_l.data[0], loss_c.data[0],
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd300_COCO_' +
                       repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
