# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block
from scipy.ndimage import label, center_of_mass
import scipy
import numpy as np

import matplotlib.pyplot as plt
from util.pos_embed import get_2d_sincos_pos_embed
import time


def visualize_patch_mask(mask, title="Patch Mask"):
    """
    Visualize a patch-level mask (e.g., 14x14) as an image.
    Args:
        mask: Tensor of shape [L], [B, L], or [H, W] (sequence length or patch grid).
        title: Title for the visualization.
    """
    # If mask is [B, L], select the first sample
    if len(mask.shape) == 2:  # [B, L]
        mask = mask[0]  # Use the first batch

    # Reshape mask to 2D if needed
    if len(mask.shape) == 1:  # [L]
        num_patches = int(mask.shape[0]**0.5)  # Assuming square patches
        mask = mask.reshape(num_patches, num_patches)  # Reshape to [H, W]

    # Convert mask to numpy for plotting
    mask = mask.cpu().numpy()  # Ensure it's on CPU

    # Plot the mask
    plt.figure(figsize=(5, 5))
    plt.imshow(mask, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) #qk_scale=None,
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)# qk_scale=None,
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask, mask_ratio):
        """
        Perform per-sample masking influenced by the mask.
        Ensure the number of patches kept aligns with mask_ratio.
        """
        preserve_object = False
        use_blob_hint=False
        N, L, D = x.shape  # batch size, sequence length, embedding dimension
        num_patches = int(L**0.5)  # Assuming square grid of patches
        #visualize_patch_mask(mask.squeeze(0).flatten(1))# Visualize the mask before patch
        # Step 1: Convert mask to patch level
        if mask is not None:
            mask = F.interpolate(mask, size=(num_patches, num_patches), mode='area')  # Resize to patch level
            mask = mask.squeeze(1).flatten(1)  # [B, L]
            mask_counts = (mask>0).float().sum(dim=1)  # Total masked pixels per sample

        else:
            mask_counts = torch.zeros(N, device=x.device)

        # Step 2: Determine number of patches to keep
        len_keep = int(L * (1 - mask_ratio))  # Fixed number of patches to keep

        # Step 3: Adjust probabilities based on mask
        noise = torch.rand(N, L, device=x.device)  # Random noise for all patches
        for i in range(noise.shape[0]):
            if mask_counts[i] >= 1:
                # Normalize the patch mask
                patch_probs = (mask[i] - mask[i].min()) / (mask[i].max() - mask[i].min() + 1e-6)  # [0, 1]

                if preserve_object:
                    # Lower noise where object is → increase chance of keeping
                    noise[i] = noise[i] + (1 - patch_probs)
                else:
                    # Higher noise where object is → increase chance of masking
                    noise[i] = noise[i] + patch_probs

                # Normalize adjusted noise
                noise[i] = (noise[i] - noise[i].min()) / (noise[i].max() - noise[i].min() + 1e-6)

        if use_blob_hint:
                # --- Blob hint strategy ---
                mask_np = mask[i].view(num_patches, num_patches).cpu().numpy()

                # Connected components (blobs)
                labeled, num_features = label(mask_np > 0.5)

                if num_features > 0:
                    blob_centers = np.array(center_of_mass(mask_np, labeled, range(1, num_features + 1)))
                    if blob_centers.ndim == 1:
                        blob_centers = blob_centers.reshape(1, -1)

                    for center_idx in blob_centers:
                        if len(center_idx) != 2:
                            continue

                        y, x_idx = int(round(center_idx[0])), int(round(center_idx[1]))
                        y = min(max(y, 0), num_patches - 1)
                        x_idx = min(max(x_idx, 0), num_patches - 1)
                        patch_id = y * num_patches + x_idx

                        # Lower noise to preserve hint patches (higher keep chance)
                        noise[i, patch_id] -= 1  # Strong hint
        #visualize_patch_mask(noise)# Visualize noise
        
        # Step 4: Sort and select patches
        ids_shuffle = torch.argsort(noise, dim=1)  # Sort noise (lower values = keep)
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # Restore order after masking
        ids_keep = ids_shuffle[:, :len_keep]  # Select indices of patches to keep

        # Step 5: Gather masked patches
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        # Step 6: Generate binary mask
        binary_mask = torch.ones([N, L], device=x.device)  # All masked by default
        binary_mask.scatter_(1, ids_keep, 0)  # Mark kept patches as 0 (unmasked) Masked are 1
        #visualize_patch_mask(binary_mask)# Visualize binary mask
        #time.sleep(60)#sleep for 1 minute
        #if not self.training:  # If in test mode, restore order immediately
        #    x_restored = torch.gather(x_masked, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        #    ids_restore = torch.arange(L, device=x.device).unsqueeze(0).expand(N, -1)  # Identity mapping
        #    return x_restored, binary_mask, ids_restore
        
        return x_masked, binary_mask, ids_restore
        


    def forward_encoder(self, x, mask=None, mask_ratio=0.75,return_attention = False):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        #store the attention map
        attention_maps = [] if return_attention else None
         
        # apply Transformer blocks
        for blk in self.blocks:
            if return_attention:
                x,attn_weights = blk(x, return_attention=True)
                attention_maps.append(attn_weights)
            x = blk(x)
        x = self.norm(x)

        if return_attention:
            return x,mask,ids_restore,attention_maps
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_input=None, mask_ratio=0.75, return_attention=False):
        if return_attention:
            latent, mask, ids_restore, attn_maps = self.forward_encoder(imgs,mask_input,mask_ratio,True)
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
            loss = self.forward_loss(imgs, pred, mask)
            return loss, pred, mask, attn_maps
        else:
            latent, mask, ids_restore = self.forward_encoder(imgs,mask_input,mask_ratio)
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
            loss = self.forward_loss(imgs, pred, mask)
            return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
