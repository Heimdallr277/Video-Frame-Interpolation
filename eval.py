

import os
import sys
import time
import copy
import shutil
import random

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import config
import utils
from loss import sequence_loss


##### Parse CmdLine Arguments #####
args, unparsed = config.get_args()
cwd = os.getcwd()
print(args)


device = torch.device('cuda' if args.cuda else 'cpu')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)


##### Load Dataset #####
train_loader, test_loader = utils.load_dataset(
    args.dataset, args.data_root, args.batch_size, args.test_batch_size, args.num_workers, args.test_mode)


##### Build Model #####

from models.VFI import VFINet
model = VFINet()

# Just make every model to DataParallel
model = torch.nn.DataParallel(model).to(device)
load_name = os.path.join('checkpoint', 'VFI_FFN', 'model_best.pth')
checkpoint = torch.load(load_name)
model.load_state_dict(checkpoint['state_dict'])

def main(args):
    losses, psnrs, ssims, lpips = utils.init_meters()
    model.eval()
    t = time.time()
    with torch.no_grad():
        for i, (images, imgpaths) in enumerate(tqdm(test_loader)):
     
            # Build input batch
            im0, im1, gt = utils.build_input(images, imgpaths, 
            is_training=False)
            
            # Forward
            out = model(im0, im1) 

            # Evaluate metrics
            utils.eval_metrics(out[-1], gt, psnrs, ssims, lpips)

                    
    # Print progress
    print('im_processed: {:d}/{:d} {:.3f}s   \r'.format(i + 1, len(test_loader), time.time() - t))
    print("Loss: %f, PSNR: %f, SSIM: %f, LPIPS: %f\n" %
          (losses[0].avg, psnrs.avg, ssims.avg, lpips.avg))
    return


if __name__ == "__main__":
    main(args)
