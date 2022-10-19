import sys,os
import logging
import argparse

import numpy as np
import torch

from eval import meval_net
from unet import UNet
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader

# dir_img = 'E:/DL_datasets/WHU Aerial imagery dataset/test/image/'
# dir_mask = 'E:/DL_datasets/WHU Aerial imagery dataset/test/label/'

# args for test
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-i', '--img_dir', dest='dir_img', type=str, default='E:/DL_datasets/WHU Aerial imagery dataset/test/image/',
                        help='Directory of the input test images')
    parser.add_argument('-m', '--mask_dir', dest='dir_mask', type=str, default='E:/DL_datasets/WHU Aerial imagery dataset/test/image/',
                        help='Directory of the input test images')
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=1, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if not args.load:
       args.load='MODEL.pth' 
    net.load_state_dict(
        torch.load(args.load, map_location=device)
    )
    logging.info(f'Image Directory:{args.dir_img}')
    logging.info(f'Mask Directory:{args.dir_mask}')
    logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    test_dataset = BasicDataset(args.dir_img, args.dir_mask, args.scale)
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=True, num_workers=0, pin_memory=True) # num_workers=0 on windows to guarantee stability

    try:
        dice_coef, prec, acc, rec = meval_net(net, test_loader, device)
        IoU = dice_coef/(2.0-dice_coef)
        print("DC: "+str(dice_coef))
        print("IoU: "+str(IoU))
        print("Precision: "+str(prec))
        print("Accuracy: "+str(acc))
        print("Recall: "+str(rec))
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
