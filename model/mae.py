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
from timm.models.vision_transformer import  Block
from util.pos_embedding import get_1d_sincos_pos_embed

class PatchEmbed_1D(nn.Module):
    """ 1D Signal to Patch Embedding
        patch_length may be the same long as embed_dim
    """ 
    def __init__(self, sig_length=2400, patch_length=40, in_chans=1, embed_dim=40, norm_layer=None, flatten=True):
        super().__init__()
        self.sig_length = sig_length
        self.patch_length = patch_length
        self.grid_size = sig_length//patch_length
        self.num_patches = self.grid_size
        self.flatten = flatten

        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Conv1d(in_chans,embed_dim,kernel_size=patch_length,stride=patch_length)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, L = x.shape
        
        assert L == self.sig_length, 'signal length does not match.'
        x = self.proj(x)
        if self.flatten:
            # x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            x = x.transpose(1,2) # BCN -> BNC
        x = self.norm(x)
        return x
    
class MaskedAutoencoderViT(nn.Module):
    
    name = 'mae1d'
    
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=2500, patch_size=50, in_chans=1,
                 embed_dim=50, depth=12, num_heads=8,
                 decoder_embed_dim=36, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,mask_ratio = 0.75, mask = 'random', all_encode_norm_layer = None):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.mask = mask
        self.patch_embed = PatchEmbed_1D(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        self.fc_norm = None
        if all_encode_norm_layer != None:
            self.fc_norm = all_encode_norm_layer(embed_dim)
        
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size, bias=True) # decoder to patch,different from 2D image
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches), cls_token=True)
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

    def patchify(self, signals):
        """
        imgs: (N, 3, H, W)
        signals:(B,1,S)
        x: (N, L, patch_size**2 *3)
        x: (B,N,L)
        """
        # p = self.patch_embed.patch_size[0]
        l = self.patch_embed.patch_length
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        # print('patchify signal :{}'.format(signals.shape))
        # print('patchify l :{}'.format(l))
        assert signals.shape[-1] % l == 0 
        # h = w = imgs.shape[2] // p
        n = signals.shape[-1] // l
        
        # print('patchify n :{}'.format(n))
        # x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = signals.reshape(shape=(signals.shape[0],n,l))
        return x

    def unpatchify(self, x):
        """
        x: (B, N, L)
        imgs: (N, 3, H, W)
        signals :(B, 1, S)
        """
        l = self.patch_embed.patch_length
        n = int(x.shape[1])
        
        signals = x.reshape(shape=(x.shape[0],1,n*l))
        return signals

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    def mean_masking(self,x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        mean_index = torch.arange(L,device=x.device)
        mean_index = torch.reshape(mean_index,(int(L/4),4))
        keep_index = mean_index[:,:int(4 * (1 - mask_ratio))].flatten()
        mask_index = mean_index[:,int(4 * (1 - mask_ratio)):].flatten()
        ids_shuffle = torch.cat((keep_index,mask_index),dim=0).repeat(N,1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore
    
    def forward_feature(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        for blk in self.blocks:
            x = blk(x)
        
        if self.fc_norm is not None:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        
        return outcome

    
    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if self.mask == 'random':
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        elif self.mask == 'mean':
            x, mask, ids_restore = self.mean_masking(x, mask_ratio)
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) #repeat cls_token
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle 回到原来的顺序
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

    def compute_loss(self, imgs, pred, mask):
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
    
    

    def forward(self, signals, mask_ratio=0.75):
        mask_ratio = self.mask_ratio
        latent, mask, ids_restore = self.forward_encoder(signals, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [B,N,L]
        loss = self.compute_loss(signals, pred, mask)
        return loss, pred, mask
    

    def forward_loss(self, signals, mask_ratio=0.75):
        mask_ratio = self.mask_ratio
        latent, mask, ids_restore = self.forward_encoder(signals, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [B,N,L]
        loss = self.compute_loss(signals, pred, mask)
        return loss






def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=25, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_signal_patch40_enc40_dec20d8b(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=40,embed_dim=40,depth=12,num_heads=10,
        decoder_embed_dim=20,decoder_depth=8,decoder_num_heads=10,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6), all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
def mae_vit_signal_patch12_enc12_dec6d8b(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=12,embed_dim=12,depth=12,num_heads=6,
        decoder_embed_dim=6,decoder_depth=8,decoder_num_heads=3,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6), all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
def mae_vit_signal_patch12_enc40_dec20d8b_m75(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=12,embed_dim=40,depth=12,num_heads=10,
        decoder_embed_dim=20,decoder_depth=8,decoder_num_heads=10,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),mask_ratio=0.75, all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
def mae_vit_signal_patch12_enc40_dec20d8b_m50(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=12,embed_dim=40,depth=12,num_heads=10,
        decoder_embed_dim=20,decoder_depth=8,decoder_num_heads=10,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),mask_ratio = 0.50, all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
def mae_vit_signal_patch12_enc40_dec20d8b_m25(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=12,embed_dim=40,depth=12,num_heads=10,
        decoder_embed_dim=20,decoder_depth=8,decoder_num_heads=10,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),mask_ratio = 0.25, all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
def mae_vit_signal_patch12_enc40_dec20d8b_m75_mean(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=12,embed_dim=40,depth=12,num_heads=10,
        decoder_embed_dim=20,decoder_depth=8,decoder_num_heads=10,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),mask_ratio=0.75,mask='mean', all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
def mae_vit_signal_patch12_enc40_dec20d8b_m50_mean(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=12,embed_dim=40,depth=12,num_heads=10,
        decoder_embed_dim=20,decoder_depth=8,decoder_num_heads=10,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),mask_ratio=0.5,mask='mean', all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
def mae_vit_signal_patch12_enc40_dec20d8b_m25_mean(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=12,embed_dim=40,depth=12,num_heads=10,
        decoder_embed_dim=20,decoder_depth=8,decoder_num_heads=10,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),mask_ratio=0.25,mask='mean', all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
def mae_vit_signal_patch24_enc80_dec40d8b_m75(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=24,embed_dim=80,depth=12,num_heads=10,
        decoder_embed_dim=40,decoder_depth=8,decoder_num_heads=10,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),mask_ratio=0.75,mask='random', all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
def mae_vit_signal_patch24_enc80_dec40d8b_m50(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=24,embed_dim=80,depth=12,num_heads=10,
        decoder_embed_dim=40,decoder_depth=8,decoder_num_heads=10,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),mask_ratio=0.5,mask='random', all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
def mae_vit_signal_patch24_enc80_dec40d8b_m25(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=24,embed_dim=80,depth=12,num_heads=10,
        decoder_embed_dim=40,decoder_depth=8,decoder_num_heads=10,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),mask_ratio=0.25,mask='random', all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
def mae_vit_signal_patch48_enc160_dec80d8b_m75(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=48,embed_dim=160,depth=12,num_heads=10,
        decoder_embed_dim=80,decoder_depth=8,decoder_num_heads=10,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),mask_ratio=0.75,mask='random', all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
def mae_vit_signal_patch48_enc160_dec80d8b_m50(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=48,embed_dim=160,depth=12,num_heads=10,
        decoder_embed_dim=80,decoder_depth=8,decoder_num_heads=10,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),mask_ratio=0.5,mask='random', all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
def mae_vit_signal_patch48_enc160_dec80d8b_m25(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=48,embed_dim=160,depth=12,num_heads=10,
        decoder_embed_dim=80,decoder_depth=8,decoder_num_heads=10,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),mask_ratio=0.25,mask='random', all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6),  **kwargs
    )
    return model


def mae_prefer_custom(winsize, patch_size, **kwargs):
    model = MaskedAutoencoderViT(
        img_size = winsize, patch_size=patch_size, embed_dim=160,depth=12,num_heads=10,
        decoder_embed_dim=80,decoder_depth=8,decoder_num_heads=10,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),mask_ratio=0.75,mask='random', all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_signal_patch40 = mae_vit_signal_patch40_enc40_dec20d8b # decoder : 20dim, 8 blocks
mae_vit_signal_patch12 = mae_vit_signal_patch12_enc12_dec6d8b # decoder : 6dim, 8 blocks
mae_vit_signal_patch12_mask75 = mae_vit_signal_patch12_enc40_dec20d8b_m75 # decoder : 6dim, 8 blocks
mae_vit_signal_patch12_mask50 = mae_vit_signal_patch12_enc40_dec20d8b_m50 # decoder : 6dim, 8 blocks
mae_vit_signal_patch12_mask25 = mae_vit_signal_patch12_enc40_dec20d8b_m25 # decoder : 6dim, 8 blocks
mae_vit_signal_patch12_mask75_mean = mae_vit_signal_patch12_enc40_dec20d8b_m75_mean # decoder : 6dim, 8 blocks
mae_vit_signal_patch12_mask50_mean = mae_vit_signal_patch12_enc40_dec20d8b_m50_mean # decoder : 6dim, 8 blocks
mae_vit_signal_patch12_mask25_mean = mae_vit_signal_patch12_enc40_dec20d8b_m25_mean # decoder : 6dim, 8 blocks
mae_vit_signal_patch24_mask75 = mae_vit_signal_patch24_enc80_dec40d8b_m75 # decoder : 6dim, 8 blocks
mae_vit_signal_patch24_mask50 = mae_vit_signal_patch24_enc80_dec40d8b_m50 # decoder : 6dim, 8 blocks
mae_vit_signal_patch24_mask25 = mae_vit_signal_patch24_enc80_dec40d8b_m25 # decoder : 6dim, 8 blocks
mae_vit_signal_patch48_mask75 = mae_vit_signal_patch48_enc160_dec80d8b_m75 # decoder : 6dim, 8 blocks
mae_vit_signal_patch48_mask50 = mae_vit_signal_patch48_enc160_dec80d8b_m50 # decoder : 6dim, 8 blocks
mae_vit_signal_patch48_mask25 = mae_vit_signal_patch48_enc160_dec80d8b_m25 # decoder : 6dim, 8 blocks