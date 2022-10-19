import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from tqdm import tqdm

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='Dirname of input images', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Dirname of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def get_allfiles(dir):
    files = os.listdir(dir)
    filtered_files=[]
    for file in files:
        if os.path.splitext(file)[1] != '.txt':#目录下包含所有非txt文件
            filtered_files.append(file)
    return filtered_files   

logging.basicConfig(level=logging.INFO,
                    filename='./predict_recording.log',
                    filemode='w')

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    args = get_args()
    # Modified by Xinbo Zhao 2020.9.9, use directory name as input and output parameters.add
    
    path_in = args.input
    if os.path.isdir(path_in[0]):
        in_files = get_allfiles(path_in[0])# image file list
    else:
        in_files=path_in
    
    out_dir = get_output_filenames(args)[0]
    if os.path.splitext(out_dir)[1]!='':#judge if out_dir is a file name with .ext
        raise Exception("Error: --output "+out_dir+" not a feasible directory")

    net = UNet(n_channels=3, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")
    try:
        with tqdm(total=len(in_files), desc='Prediction', unit='per image', leave=True) as pbar:
            for i, fn in enumerate(in_files):
                img_name = path_in[0]+'/'+fn # usable path of imgs
                logging.info("\nPredicting image {} ...".format(img_name))

                img = Image.open(img_name)

                mask = predict_img(net=net,
                                full_img=img,
                                scale_factor=args.scale,
                                out_threshold=args.mask_threshold,
                                device=device)

                if not args.no_save:
                    if not os.path.exists(out_dir):
                        os.mkdir(out_dir)
                    out_fn = out_dir+'/'+fn.split('.')[0] + '_mask.tif'
                    result = mask_to_image(mask)
                    result.save(out_fn)
                    logging.info("Mask saved to {}".format(out_fn))

                if args.viz:
                    logging.info("Visualizing results for image {}, close to continue ...".format(fn))
                    plot_img_and_mask(img, mask)

                pbar.update()
    except KeyboardInterrupt:
        logging.info('Keyboard interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
