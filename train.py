
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
from loss import train_loss
import math

##### Parse CmdLine Arguments #####
args, unparsed = config.get_args()
cwd = os.getcwd()
print(args)


##### TensorBoard & Misc Setup #####
if args.mode != 'test':
    writer = SummaryWriter('logs/%s' % args.exp_name)

device = torch.device('cuda' if args.cuda else 'cpu')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)


##### Load Dataset #####
train_loader, val_loader = utils.load_dataset(
    args.dataset, args.data_root, args.batch_size, args.val_batch_size, args.num_workers)


##### Build Model #####
if args.model.lower() == 'vfi':
    from models.VFI import VFINet
    print('Building model: VFINet')
    model = VFINet(args)
else:
    raise NotImplementedError("Unknown model!")
# Just make every model to DataParallel
model = torch.nn.DataParallel(model).to(device)
#print(model)

##### Define Loss & Optimizer #####
criterion = train_loss(args)


from torch.optim import Adam
optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
print('# of parameters: %d' % sum(p.numel() for p in model.parameters()))


# If resume, load checkpoint: model + optimizer
best_psnr = 0
if args.resume:
    best_psnr = utils.load_checkpoint(args, model, optimizer)
    
# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=4, verbose=True)


# Initialize LPIPS model if used for evaluation
# lpips_model = utils.init_lpips_eval() if args.lpips else None
lpips_model = None




def train(args, epoch):
    losses, psnrs, ssims, lpips = utils.init_meters()
    model.train()
    criterion.train()
    # torch.autograd.set_detect_anomaly(True)
    t = time.time()
    for i, (images, imgpaths) in enumerate(train_loader):
        # Build input batch
        im0, im1, gt = utils.build_input(images, imgpaths)
        # Forward
        optimizer.zero_grad()
        out = model(im0, im1)
        loss = criterion(out, gt)
        # Save loss values
        for j in range(len(losses)):
            losses[j].update(loss[j].item())

        # Backward (+ grad clip) 
        # with torch.autograd.detect_anomaly():
        loss[0].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        # Calc metrics & print logs
        if i % args.log_iter == 0 and i > 0:
            utils.eval_metrics(out[0], gt, psnrs, ssims, lpips, lpips_model)
            print('Train Epoch: {} [{}/{}]\tLoss_total: {:.6f}\tLoss_GT: {:.6f}\tLoss_Perceptual: {:.6f}\tLoss_Reconstru: {:.6f}\tPSNR: {:.4f}\tTime({:.2f})'.format(
                epoch, i, len(train_loader), losses[0].avg, losses[1].avg, losses[2].avg, losses[3].avg, psnrs.avg, time.time() - t))
            
            # Log to TensorBoard
            utils.log_tensorboard(writer, losses, psnrs.avg, ssims.avg, lpips.avg, optimizer.param_groups[-1]['lr'], epoch * len(train_loader) + i)

            # Reset metrics
            losses, psnrs, ssims, lpips = utils.init_meters()
            t = time.time()

def val(args, epoch):
    print('Evaluating for epoch = %d' % epoch)
    losses, psnrs, ssims, lpips = utils.init_meters()
    model.eval()
    criterion.eval()

    save_dir = os.path.join('checkpoint', args.exp_name) 
    utils.makedirs(save_dir)
    t = time.time()
    with torch.no_grad():
        for i, (images, imgpaths) in enumerate(tqdm(val_loader)):
     
            # Build input batch
            im0, im1, gt = utils.build_input(images, imgpaths, is_training=False)
            out = model(im0, im1)
            loss = criterion(out, gt)
            for j in range(len(losses)):
                losses[j].update(loss[j].item())
            utils.eval_metrics(out[0], gt, psnrs, ssims, lpips)

    print('im_processed: {:d}/{:d} {:.3f}s   \r'.format(i + 1, len(val_loader), time.time() - t))
    print("Loss: %f, PSNR: %f, SSIM: %f, LPIPS: %f\n" %
          (losses[0].avg, psnrs.avg, ssims.avg, lpips.avg))

    # Save psnr & ssim
    save_fn = os.path.join('checkpoint', args.exp_name, 'results.txt')
    with open(save_fn, 'a') as f:
        f.write("Epoch: %d, PSNR: %f, SSIM: %f, LPIPS: %f\n" %
                (epoch, psnrs.avg, ssims.avg, lpips.avg))

    # Log to TensorBoard
    utils.log_tensorboard(writer, losses, psnrs.avg, ssims.avg, lpips.avg,
        optimizer.param_groups[-1]['lr'], (epoch+1) * len(train_loader), mode='test')

    return losses[0].avg, psnrs.avg, ssims.avg, lpips.avg           


            # # Log examples that have bad performance
            # if (ssims.val < 0.9 or psnrs.val < 25) and epoch > 50:
            #     print(imgpaths)
            #     print("\nLoss: %f, PSNR: %f, SSIM: %f, LPIPS: %f" %
            #           (losses[0].val, psnrs.val, ssims.val, lpips.val))
            #     print(imgpaths[1][-1])

            # # Save result images
            # if ((epoch + 1) % 1 == 0 and i < 20) or args.mode == 'test':
            #     savepath = os.path.join('checkpoint', args.exp_name, save_folder)

            #     for b in range(images[0].size(0)):
            #         paths = imgpaths[1][b].split('/')
            #         fp = os.path.join(savepath, paths[-3], paths[-2])
            #         if not os.path.exists(fp):
            #             os.makedirs(fp)
            #         # remove '.png' extension
            #         fp = os.path.join(fp, paths[-1][:-4])
            #         utils.save_image(out[0][b], "%s.png" % fp)
                    
 


""" Entry Point """
def main(args):
    global best_psnr
    
    # best_psnr = 0
    for epoch in range(args.start_epoch, args.max_epoch):
        # run training
        train(args, epoch)
        # run test
        test_loss, psnr, _, _ = val(args, epoch)

        # update optimizer policy
        scheduler.step(test_loss)

        # save checkpoint
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)
        utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_psnr': best_psnr
        }, is_best, args.exp_name)

      


if __name__ == "__main__":
    main(args)