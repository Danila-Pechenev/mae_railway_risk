# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
import matplotlib.pyplot as plt


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, masks) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True) if masks is not None else None
        
        #Print samples et mask if batch size=1 TO DELETE
        #plt.imshow(samples.squeeze(0).cpu().permute(1,2,0).numpy())
        #plt.axis('off')
        #plt.show()
        
        #plt.imshow(masks.squeeze(0).cpu().permute(1,2,0).numpy())
        #plt.axis('off')
        #plt.show() 

        with torch.cuda.amp.autocast():
            # Always pass the mask to the model, let the model handle None masks internally
            loss, _, _ = model(samples, mask_input=masks, mask_ratio=args.mask_ratio, preserve_object=args.preserve_object, blob_hint=args.blob_hint)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_reconstruction(data_loader, model, device, num_samples=5):
    """
    Evaluate the reconstruction quality of a pretrained MAE model.
    """
    metric_logger = misc.MetricLogger(delimiter="  ")
    model.eval()

    total_mse, total_psnr, total_ssim = 0, 0, 0
    num_images = 0

    for batch_idx, (images, masks) in enumerate(data_loader):
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass through the model
        loss, pred, mask = model(images, masks)
        reconstructed = model.unpatchify(pred)  # Reconstruct images
        
        # **Ensure Shape Compatibility**
        images_np = images.cpu().numpy()
        reconstructed_np = reconstructed.cpu().numpy()

        # Compute Reconstruction Metrics
        mse_loss = F.mse_loss(reconstructed, images)
        psnr = 10 * torch.log10(1 / mse_loss)
        
        # **Fix SSIM Calculation**
        try:
            ssim_val = ssim(
                images_np.transpose(0, 2, 3, 1),  # Convert (B, C, H, W) -> (B, H, W, C)
                reconstructed_np.transpose(0, 2, 3, 1),
                data_range=1,
                channel_axis=-1,  # Ensure multichannel works
                win_size=5  # Smaller window to avoid "win_size exceeds image extent"
            )
        except ValueError as e:
            print(f"SSIM Computation Error: {e}")
            ssim_val = 0  # Assign zero if SSIM fails
        
        total_mse += mse_loss.item()
        total_psnr += psnr.item()
        total_ssim += ssim_val
        num_images += 1

        # Show first few reconstructions
        #if batch_idx < num_samples:
        #    visualize_reconstruction(images, reconstructed, mask, batch_idx)

    # Print final evaluation scores
    print(f"Reconstruction Metrics - MSE: {total_mse / num_images:.4f}, PSNR: {total_psnr / num_images:.2f}, SSIM: {total_ssim / num_images:.4f}")

    return {
        "mse": total_mse / num_images,
        "psnr": total_psnr / num_images,
        "ssim": total_ssim / num_images
    }