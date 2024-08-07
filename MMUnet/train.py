import argparse
import os
from collections import OrderedDict
from glob import glob
import datetime
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from albumentations.core.composition import Compose, OneOf
import albumentations as albu
from nets.MMUnet import MMUnet
import losses
from dataset import Dataset
from metrics import calculate
from utils import AverageMeter, str2bool


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=400, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int, metavar='N', help='mini-batch size (default: 16)')

    parser.add_argument('--arch', '-a', metavar='ARCH', default='MMDUnet')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int, help='input channels')
    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
    parser.add_argument('--input_w', default=256, type=int, help='image width')
    parser.add_argument('--input_h', default=256, type=int, help='image height')

    parser.add_argument('--loss', default='BCEDiceLoss', choices=['BCEWithLogitsLoss', 'BCEDiceLoss'],
                        help='loss: ' + ' | '.join(['BCEWithLogitsLoss', 'BCEDiceLoss']) + ' (default: BCEDiceLoss)')

    parser.add_argument('--dataset', default='BUSI', help='dataset name')
    parser.add_argument('--img_ext', default='.png', help='image file extension')
    parser.add_argument('--mask_ext', default='.png', help='mask file extension')

    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD', 'AdamW'],
                        help='loss: ' + ' | '.join(['Adam', 'SGD']) + ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-5, type=float, help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool, help='nesterov')

    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float, help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int, metavar='N', help='early stopping (default: -1)')

    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )
    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer, device):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.to(device)
        target = target.to(device)

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output, target in zip(outputs, target):
                loss += criterion(output, target)
            loss /= len(outputs)
            iou, F1 = calculate(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou, F1 = calculate(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion, device):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                   'F1': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.to(device)
            target = target.to(device)

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output, target in zip(outputs, target):
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, F1 = calculate(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, F1 = calculate(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['F1'].update(F1, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('F1', avg_meters['F1'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('F1', avg_meters['F1'].avg)
                        ])


def main():
    config = vars(parse_args())

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_withDS_%s_%s_%s' % (config['dataset'], config['arch'], config['optimizer'], config['scheduler'], datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        else:
            config['name'] = '%s_%s_woDS_%s_%s_%s' % (config['dataset'], config['arch'], config['optimizer'], config['scheduler'], datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    writer = SummaryWriter(log_dir='./models/%s' % config['name'])

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.BCEDiceLoss().cuda()

    cudnn.benchmark = True

    # create model
    if config['arch'] == 'MMDUnet':
        model = MMUnet(config['input_channels'], config['num_classes'])
    else:
        raise NotImplementedError

    device = torch.device("cuda:{}".format(0))
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])

    # optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    # scheduler
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    train_transform = Compose([
        albu.RandomRotate90(),
        albu.Flip(),
        albu.Resize(config['input_h'], config['input_w']),
        albu.Normalize(),
    ])

    val_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
        albu.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_F1', []),
    ])

    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        train_log = train(config, train_loader, model, criterion, optimizer, device)

        writer.add_scalar('Loss/Train', train_log['loss'], epoch)
        writer.add_scalar('IOU/Train', train_log['iou'], epoch)

        val_log = validate(config, val_loader, model, criterion, device)

        writer.add_scalar('Loss/Validation', val_log['loss'], epoch)
        writer.add_scalar('IOU/Validation', val_log['iou'], epoch)
        writer.add_scalar('F1/Validation', val_log['F1'], epoch)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f - val_F1 %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou'], val_log['F1']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_F1'].append(val_log['F1'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' % config['name'], index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            with open('models/%s/update_best_index.txt' % config['name'], 'a') as file:
                file.write('epoch:' + str(epoch) + '\n')
                file.write('iou:' + str(val_log['iou']) + '\n')
                file.write('F1:' + str(val_log['F1']) + '\n')
            torch.save(model.state_dict(), 'models/%s/model.pth' % config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()
    writer.close()

if __name__ == "__main__":
    main()