# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import shutil
from statistics import mean
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import distributed_all_gather
from monai.data import decollate_batch
import wandb
from patchify import patchify, unpatchify
import nibabel as nib
#from monai.visualize import blend_images, matshow3d, plot_2d_or_3d_image
import sys
from einops import rearrange

#####################################################################
# ##########################plot_function##############################
# def vis_image_patches(data):
#     #data=data[0]
#         fig = plt.figure()
#         matshow3d(data, fig=fig, title="data")
#         plt.show()

###### masking ###########
def patchify(imgs, patch_size):
    """
    imgs: (N, 1, H, W,D)
    x: (N, L, patch_size**3 *1)
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = d= imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p,d,p))
    x = torch.einsum('nchpwqdr->nhwdpqrc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w * d, p ** 3 * 1))
    return x

def unpatchify(x, patch_size):
    """
    x: (N, L, patch_size**3 *1)
    imgs: (N, 1, H, W,D)
    """
    p = patch_size
    #h = w = d = int(x.shape[1] ** (1./3.))
    h = w = d = math.ceil(x.shape[1] ** (1. / 3.))
    assert h * w * d == x.shape[1]
    x = x.reshape(shape=(x.shape[0], h, w,d, p, p,p, 1))
    x = torch.einsum('nhwdpqrc->nchpwqdr', x)
    imgs = x.reshape(shape=(x.shape[0], 1, h * p, w * p, d * p))
    return imgs

def mask(img, mask_ratio, patch_size):
    """
    To create masking patches in the img given a mask-ratio
    :param mask_ratio: Image area to be masked (between 0-1)
    :param patch_size: Size of the image patches.
    :return: returns the masked image
    """
    patches = patchify(img, patch_size=patch_size)
    N, L, D = patches.shape
    len_mask = int(L * (mask_ratio))
    noise = torch.rand(N, L)  # noise in [0, 1]
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    patches[:, ids_shuffle[0][0:len_mask], :] = 0
    masked_img = unpatchify(patches, patch_size=patch_size)
    masked_img = masked_img
    return masked_img

###### shuffling ###########


def shuffle(img): #[4,1,96,96,96]
    inputs = img.permute(0, 4, 1, 2, 3) #[4,96,1,96,96]
    inputs_b = rearrange(inputs, 'b (n d) c h w -> b n d c h w', n=6) #[4, 6, 16, 1, 96, 96]
    B, N, D, C, H, W = inputs_b.shape

    indices = np.random.permutation(N)
    shuffled_inputs = inputs_b[:, indices] #[4, 6, 16, 1, 96, 96]shuffled
    

    inputs_sh = rearrange(shuffled_inputs, 'b n d c h w -> b (n d) c h w', n=6)
    #inputs_final = inputs_sh.transpose(1, 2).contiguous()
    inputs_final =inputs_sh.permute(0,2,3,4,1)
    return inputs_final,indices


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None):
    to_shuffle =False
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data,to_shuffle=False)
                else:
                    logits = model(data,to_shuffle=False)
            if not logits.is_cuda:
                target = target.cpu()
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc = acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc = acc.cuda(args.rank)

            if args.distributed:
                acc_list = distributed_all_gather([acc], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
                avg_acc = np.mean([np.nanmean(l) for l in acc_list])

            else:
                acc_list = acc.detach().cpu().numpy()
                avg_acc = np.mean([np.nanmean(l) for l in acc_list])
            if args.rank == 0:
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "acc",
                    avg_acc,
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
            wandb.log({"avg_accuracy": avg_acc})
    return avg_acc


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)

def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    to_shuffle = args.shufflemask_pretrain
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            if to_shuffle:
                sh_data,indices =shuffle(data)
                mask_sh_data =mask(sh_data,args.mask_ratio,args.mask_patch_size)
                rec_img,t1,t2,t3,t4 =model(mask_sh_data,to_shuffle=True)
                loss_rec = torch.nn.MSELoss()(rec_img,sh_data)
                wandb.log({"loss_rec": loss_rec})
                

                B, T = t1.shape[:2]
                t_label = torch.LongTensor(list(indices)).unsqueeze(0).repeat(B,1).cuda()
                loss_cls1 = torch.nn.CrossEntropyLoss()(t1.view(B*T,-1), t_label.view(-1))
                loss_cls2 = torch.nn.CrossEntropyLoss()(t2.view(B*T,-1), t_label.view(-1))
                loss_cls3 = torch.nn.CrossEntropyLoss()(t3.view(B*T,-1), t_label.view(-1))
                loss_cls4 = torch.nn.CrossEntropyLoss()(t4.view(B*T,-1), t_label.view(-1))
                loss_cls =loss_cls1+loss_cls2+loss_cls3+loss_cls4
                wandb.log({"loss_cls": loss_cls})

                loss = args.alpha1* loss_rec + args.alpha2 *loss_cls

            else:  
                logits = model(data,to_shuffle=False)
                loss = loss_func(logits, target)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
        wandb.log({"loss": run_loss.avg})
    for param in model.parameters():
        param.grad = None
    return run_loss.avg

def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    to_shuffle=args.shufflemask_pretrain
    if to_shuffle:
        print("shuffle and mask pretraining")
    else:
        print("Normal Training")
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )
            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc",
                    val_avg_acc,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint and epoch%100==0 or epoch==args.max_epochs-1:
                 save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_epoch" + f'{epoch}.pt')
            if scheduler is not None:
                scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max

