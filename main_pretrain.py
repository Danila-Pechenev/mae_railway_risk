# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import random
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import to_tensor

import timm

#assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import albumentations as A
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch, evaluate_reconstruction

SEMANTIC_CLASS_TO_ID = {
    'road': 0,
    'sidewalk': 1,
    'construction': 2,
    'tram-track': 3,
    'fence': 4,
    'pole': 5,
    'traffic-light': 6,
    'traffic-sign': 7,
    'vegetation': 8,
    'terrain': 9,
    'sky': 10,
    'human': 11,
    'rail-track': 12,
    'car': 13,
    'truck': 14,
    'trackbed': 15,
    'on-rails': 16,
    'rail-raised': 17,
    'rail-embedded': 18,
}

def add_weight_decay(model, weight_decay, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in skip_list):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay},
    ]
        
class CustomDataset(Dataset):
    # Return an image of size 224,224, an optional masking-guidance map,
    # and an optional target-class mask used for the reconstruction loss.
    def __init__(self, image_dir, mask_dir=None, semantic_mask_dir=None,
                 mask_class_ids=None, target_class_ids=None, input_size=224, transform=None, normalize=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.semantic_mask_dir = semantic_mask_dir
        self.mask_class_ids = tuple(mask_class_ids or [])
        self.target_class_ids = tuple(target_class_ids or [])
        self.input_size = input_size
        self.transform = transform
        self.normalize = normalize

        # Collect all image paths from subdirectories
        self.image_paths = []
        for root, _, files in os.walk(image_dir):
            for f in files:
                if f.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.image_paths.append(os.path.join(root, f))
        #Sort all images in order
        self.image_paths = sorted(self.image_paths)
        #random.shuffle(self.image_paths) #Randomize for testing

        # Collect matching mask paths if provided
        self.mask_paths = self._collect_mask_paths(mask_dir, "guidance masks")
        self.semantic_mask_paths = self._collect_mask_paths(semantic_mask_dir, "semantic masks")

        # Print findings
        print(f"Found {len(self.image_paths)} images.")
        if self.mask_paths:
            print(f"Found {len(self.mask_paths)} masks.")
        if self.semantic_mask_paths:
            print(f"Found {len(self.semantic_mask_paths)} semantic masks.")

    def _collect_mask_paths(self, mask_dir, description):
        if mask_dir:
            print(f"{description.capitalize()} are loaded\n")
            mask_paths = {
                os.path.splitext(f)[0]: os.path.join(mask_dir, f)
                for root, _, files in os.walk(mask_dir) for f in files
                if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            }
            print(len(mask_paths))
            if len(mask_paths) == 0:
                print(f"No {description} found, using default zeros")
                return None
            return mask_paths
        return None

    def _load_mask(self, mask_paths, image_path, class_ids=None):
        if not mask_paths:
            return None

        mask_key = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = mask_paths.get(mask_key)
        if not mask_path:
            return None

        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((self.input_size, self.input_size), Image.NEAREST)
        if class_ids is None:
            return mask

        mask_np = np.array(mask)
        mask_np = np.isin(mask_np, class_ids).astype(np.uint8) * 255
        return Image.fromarray(mask_np)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        # image_path = "../../data/images/2vsall/train/risky/rs02961.jpg" # one image for testing
        image = Image.open(image_path).convert('RGB')
        # Load corresponding masks
        image_data = None
        guidance_mask = None
        semantic_guidance_mask = None
        target_mask = None
        
        guidance_mask = self._load_mask(self.mask_paths, image_path)
        if self.mask_class_ids:
            semantic_guidance_mask = self._load_mask(
                self.semantic_mask_paths, image_path, class_ids=self.mask_class_ids)
        if self.target_class_ids:
            target_mask = self._load_mask(
                self.semantic_mask_paths, image_path, class_ids=self.target_class_ids)
        
        # Apply transforms
        if self.transform:
            image_data = self.transform(image=np.array(image))
            image = to_tensor(Image.fromarray(image_data['image']))
            #Image.fromarray(image_data['image']).show(title="Image")#Vizualise image
        else:
            image = to_tensor(image)  # Convert to tensor if no transform is applied

        # If masks are found and transform is set, replay the exact same crop/flip.
        if guidance_mask is not None:
            if self.transform:
                guidance_mask_data = A.ReplayCompose.replay(image_data['replay'], image=np.array(guidance_mask))
                guidance_mask = to_tensor(Image.fromarray(guidance_mask_data['image']))
            else:
                guidance_mask = to_tensor(guidance_mask)
        else:
            guidance_mask = torch.zeros((1, image.shape[1], image.shape[2]))

        if semantic_guidance_mask is not None:
            if self.transform:
                semantic_guidance_mask_data = A.ReplayCompose.replay(
                    image_data['replay'], image=image_data['image'], mask=np.array(semantic_guidance_mask))
                semantic_guidance_mask = to_tensor(Image.fromarray(semantic_guidance_mask_data['mask']))
            else:
                semantic_guidance_mask = to_tensor(semantic_guidance_mask)
            guidance_mask = torch.maximum(guidance_mask, semantic_guidance_mask)

        if target_mask is not None:
            if self.transform:
                target_mask_data = A.ReplayCompose.replay(
                    image_data['replay'], image=image_data['image'], mask=np.array(target_mask))
                target_mask = to_tensor(Image.fromarray(target_mask_data['mask']))
            else:
                target_mask = to_tensor(target_mask)
        else:
            target_mask = torch.zeros((1, image.shape[1], image.shape[2]))

        #Normalize only the image
        if self.normalize:
            image = self.normalize(image)

        #here the returning shape are :
        # image : channel,size,size
        # guidance_mask : 1,size,size
        # target_mask : 1,size,size
        return image, guidance_mask, target_mask



def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    
    parser.add_argument('--mask_importance', default=1.0, type=float,
                        help='Masking importance (sensitivity of the guidance mask to have an impact on the masking mask).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    
    parser.set_defaults(norm_pix_loss=False)
    
    parser.add_argument('--preserve_object', action='store_true',
                    help='If set, reduces the chance to mask object of interest (e.g. pedestrian)')

    parser.add_argument('--blob_hint', action='store_true',
                    help='If set, enables hinting strategy that leaves a patch visible per risk blob')


    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--save_freq', default=0, type=int,
                        help='save numbered checkpoints every N epochs; '
                             'set to 0 to disable numbered checkpoints; '
                             'checkpoint-last.pth is still updated after every epoch')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', required=True, type=str,
                        help='dataset path')
    parser.add_argument('--mask_path', default=None, type=str,
                        help='path to precomputed masking-guidance masks (optional)')
    parser.add_argument('--semantic_mask_path', default=None, type=str,
                        help='path to semantic segmentation masks; used when --mask_class and/or '
                             '--target_class is specified')
    parser.add_argument('--mask_class', nargs='+', default=None,
                        choices=list(SEMANTIC_CLASS_TO_ID.keys()),
                        help='one or more semantic classes to use as masking guidance; '
                             'specify them as a space-separated list; omit this option to disable '
                             'class-based masking guidance')
    parser.add_argument('--target_class', nargs='+', default=None,
                        choices=list(SEMANTIC_CLASS_TO_ID.keys()),
                        help='one or more semantic classes to emphasize in the reconstruction loss; '
                             'specify them as a space-separated list; omit this option to disable '
                             'target-class emphasis')
    parser.add_argument('--target_loss_weight', default=1.0, type=float,
                        help='extra weight applied to masked patches containing any selected target class')
    parser.add_argument('--target_loss_mode', default='proportional', choices=['binary', 'proportional'],
                        help='how the combined semantic target mask is converted into patch-level loss weights: '
                             '"binary" upweights any masked patch containing at least one selected target-class pixel, '
                             'while "proportional" upweights masked patches in proportion to the fraction of '
                             'selected target-class pixels they contain')
    parser.add_argument('--unsorted_data', action='store_true',
                        help='Indicates that the data is not sorted into train val test or classes')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    if args.save_freq < 0:
        raise ValueError('--save_freq must be >= 0')
    args.mask_class = list(dict.fromkeys(args.mask_class or []))
    args.target_class = list(dict.fromkeys(args.target_class or []))
    mask_class_ids = [SEMANTIC_CLASS_TO_ID[mask_class] for mask_class in args.mask_class]
    target_class_ids = [SEMANTIC_CLASS_TO_ID[target_class] for target_class in args.target_class]
    args.target_loss_proportional = args.target_loss_mode == 'proportional'
    use_target_loss = bool(args.target_class) and args.semantic_mask_path is not None and args.target_loss_weight > 0
    effective_target_loss_weight = args.target_loss_weight if use_target_loss else 0.0

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed #+ misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = False #DONT FORGET TO CHANGE THIS #True
    cudnn.deterministic = True #DONT FORGET TO CHANGE THIS #delete it

    # simple augmentation
    #transform_train = transforms.Compose([
    #        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
    #        transforms.RandomHorizontalFlip(),
    #        transforms.ToTensor(),
    #        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    #augmentation with albumentations
    #albumentations allow to compose tranforms that can be recorded whom will impact multiple inputs (when random crop and random flip)
    transform_train = A.ReplayCompose([
        A.RandomResizedCrop((args.input_size,args.input_size), scale=(0.2, 1.0), interpolation=3),
        A.HorizontalFlip()])
    
    normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Original dataset_train initialization (commented out)
    #dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    image_dir_data = None
    #Deal with unsorted data
    if args.unsorted_data:
        image_dir_data = args.data_path
    else:
        image_dir_data = os.path.join(args.data_path, 'train')
    #Create the dataset for training
    dataset_train = CustomDataset(
    image_dir=image_dir_data,
    mask_dir=args.mask_path,
    semantic_mask_dir=args.semantic_mask_path,
    mask_class_ids=mask_class_ids,
    target_class_ids=target_class_ids,
    input_size=args.input_size,
    transform=transform_train,
    normalize=normalize_transform
    )
    #print(dataset_train)

    if args.mask_class and args.semantic_mask_path:
        print(f"Semantic masking guidance enabled for classes={args.mask_class}")
        if args.mask_path:
            print("Combining semantic masking guidance with mask_path guidance using pixelwise maximum.")
    elif args.mask_class:
        print(
            "Warning: mask_class was provided without semantic_mask_path. "
            "Class-based masking guidance will be disabled."
        )

    if use_target_loss:
        print(
            f"Target reconstruction loss enabled with weight={args.target_loss_weight} "
            f"for classes={args.target_class} "
            f"(mode={args.target_loss_mode})"
        )
    elif args.target_class and args.semantic_mask_path:
        print(
            "Warning: target classes and semantic masks were provided, but target_loss_weight <= 0. "
            "The training will behave like the baseline loss."
        )
    elif args.target_class or args.semantic_mask_path:
        print(
            "Warning: target-class emphasis requires both semantic_mask_path and target_class. "
            "The training will behave like the baseline loss."
        )

    args.target_loss_weight = effective_target_loss_weight

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    print(f"Number of batches : {len(data_loader_train)}")

    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = add_weight_decay(model_without_ddp, args.weight_decay)

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)


    #Eval only
    #Random mask can be used here
    if args.eval:
        #Build val dataloader
        if args.unsorted_data:
            image_dir_data = args.data_path
        else:
            image_dir_data = os.path.join(args.data_path, 'val')
        dataset_val = CustomDataset(
        image_dir=image_dir_data,
        mask_dir=args.mask_path,
        semantic_mask_dir=args.semantic_mask_path,
        mask_class_ids=mask_class_ids,
        target_class_ids=target_class_ids,
        input_size=args.input_size,
        transform=transform_train,
        normalize=normalize_transform
        )
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        
        eval_stats = evaluate_reconstruction(
            data_loader_val,
            model,
            device,
            target_loss_weight=effective_target_loss_weight,
            target_loss_proportional=args.target_loss_proportional,
            mask_ratio=args.mask_ratio,
            preserve_object=args.preserve_object,
            blob_hint=args.blob_hint,
        )
        print(f"Evaluation Completed:\n{eval_stats}")
        exit(0)

    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        keep_epoch_checkpoint = args.save_freq > 0 and ((epoch + 1) % args.save_freq == 0)
        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, keep_last=True,
                keep_epoch_checkpoint=keep_epoch_checkpoint)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
