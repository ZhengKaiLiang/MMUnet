import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dataset import Dataset
from metrics import calculate
from utils import AverageMeter
from albumentations import RandomRotate90,Resize
import time
from nets.MMDUnet import MMDUnet


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='model name',
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    print("=> creating model %s" % config['arch'])

    if config['arch'] == 'MMDUnet':
        model = MMDUnet(config['input_channels'], config['num_classes'])
    else:
        raise NotImplementedError
    device = torch.device("cuda:{}".format(0))
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    model.load_state_dict(torch.load('models/%s/model.pth' % config['name']))
    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    F1_avg_meter = AverageMeter()
    gput = AverageMeter()
    cput = AverageMeter()

    count = 0

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.to(device)
            target = target.to(device)
            #model = model.to(device)
            # compute output
            output = model(input)

            iou, F1 = calculate(output, target)
            iou_avg_meter.update(iou, input.size(0))
            F1_avg_meter.update(F1, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            output[output >= 0.5] = 1
            output[output < 0.5] = 0

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % iou_avg_meter.avg)
    print('F1: %.4f' % F1_avg_meter.avg)
    with open('outputs/%s/iou_dice.txt' % config['name'], 'a') as file:
        file.write('iou:' + str(iou_avg_meter.avg) + '\n')
        file.write('F1:' + str(F1_avg_meter.avg) + '\n')

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
