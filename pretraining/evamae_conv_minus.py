# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed

################################################################################

class ElevationConv(torch.nn.Module):
    
    def __init__(self,
                 elev_in_ch,
                 out_channels, 
                 kernel_size = 3,
                 padding = 1,
                 bias = False,
                 padding_mode = 'replicate'):
        
        
        super(ElevationConv, self).__init__()
        
        self.elev_in_ch = elev_in_ch
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias
        self.padding_mode = padding_mode
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.conv_layer_elev = torch.nn.Conv2d(self.elev_in_ch,
                                              self.out_channels, 
                                              kernel_size = self.kernel_size, 
                                              stride=1, 
                                              padding=self.padding, 
                                              padding_mode=self.padding_mode, 
                                              device = self.device)
    
    def forward(self, elevation_data):
        elev_conv = self.conv_layer_elev(elevation_data)
        
        return elev_conv
    
class conv(nn.Module):
    
    def __init__(self, elev_in_ch, out_ch, kernel_size, normalize = True, activate = True):
        
        super(conv, self).__init__()
        
        self.first_Conv = ElevationConv(elev_in_ch, out_ch, kernel_size)
        self.activate = activate

        if self.activate:
            self.first_activation_img = nn.ReLU()

    def forward(self, h):

        h = self.first_Conv(h)

        if self.activate:
            h = self.first_activation_img(h)

        return h

class inconv(nn.Module):
    def __init__(self, elev_in_ch, out_ch, kernel_size):
        super(inconv, self).__init__()
        self.conv = conv(elev_in_ch, out_ch, kernel_size)

    def forward(self, h):
        h = self.conv(h)
        return h

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, conv_kernel_size=3, use_double_conv=False):
        super().__init__()

        self.in_c = in_chans

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.patch_embed_dem = PatchEmbed(img_size, patch_size, 1, embed_dim)
        self.patch_embed_dem_dec = PatchEmbed(img_size, patch_size, 1, decoder_embed_dim)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        self.use_double_conv = use_double_conv

        #########
        # Conv layer specifics
        self.inc = inconv(1, 1, conv_kernel_size)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.mask = None
        self.ids_restore = None
        self.ids_keep = None

        # saugat: for gating mechanism
        self.dem_sigmoid = torch.nn.Sigmoid()

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

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w_dem = self.patch_embed_dem.proj.weight.data
        torch.nn.init.xavier_uniform_(w_dem.view([w_dem.shape[0], -1]))

        w_dem_dec = self.patch_embed_dem_dec.proj.weight.data
        torch.nn.init.xavier_uniform_(w_dem_dec.view([w_dem_dec.shape[0], -1]))

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

    def patchify(self, imgs, p, c):
        """
        imgs: (N, C, H, W)
        p: Patch embed patch size
        c: Num channels
        x: (N, L, patch_size**2 *C)
        """
        # p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        # c = self.in_c
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        return x

    def unpatchify(self, x, p, c):
        """
        x: (N, L, patch_size**2 *C)
        p: Patch embed patch size
        c: Num channels
        imgs: (N, C, H, W)
        """
        # c = self.in_c
        # p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

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

        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)

        self.mask = mask
        self.ids_restore = ids_restore
        self.ids_keep = ids_keep

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        # for blk in self.blocks:
        #     x = blk(x)
        # x = self.norm(x)

        return x, mask, ids_restore
    
    def forward_encoder_dem(self, x, mask_ratio):
        # print("inside forward_encoder_dem")
        # x is (N, C, H, W)
        b, c, h, w = x.shape # saugat: b is batch size

        # print("x_shape_before: ", x.shape)

        x = self.inc(x)

        x = self.patch_embed_dem(x)
        # print("x_shape_after_patch_embed: ", x.shape)

        _, L, D = x.shape

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        # x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)

        x_orig = x.clone()
        x = torch.gather(x.view(b, -1, D), dim=1, index=self.ids_keep.unsqueeze(-1)).repeat(1, 1, D)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # # apply Transformer blocks
        # for blk in self.blocks:
        #     x = blk(x)
        # x = self.norm(x)

        return x, self.mask, self.ids_restore

    def forward_decoder(self, x, dem, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        dem = self.inc(dem)
        
        dem = self.patch_embed_dem_dec(dem)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # dem = self.dem_sigmoid(dem) # gating
        # x_ = x_ * dem # fusing
        x_ = x_ + dem # fusing
        
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        # target = imgs[:, :3, :, :]
        # pred = self.unpatchify(pred, self.patch_embed.patch_size[0], self.in_c)
        # pred = self.patchify(pred[:, :3, :, :], self.patch_embed.patch_size[0], 3)
        # target = self.patchify(target, self.patch_embed.patch_size[0], 3)

        # print("imgs_shape: ", imgs.shape)
        target = self.patchify(imgs, self.patch_embed.patch_size[0], self.in_c)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        # print("target_1: ", target.shape)

        N, L, _ = target.shape
        target = target.view(N, L, self.in_c, -1)  # (N, L, C, p^2)
        target = torch.einsum('nlcp->nclp', target)  # (N, C, L, p^2)

        N_p, L_p, _ = pred.shape
        pred = pred.view(N_p, L_p, self.in_c, -1)  # (N, L, C, p^2)
        # pred = pred.einsum('nlcp->nclp', target)  # (N, C, L, p^2)
        pred = pred.permute(0, 2, 1, 3)

        # print("target_2: ", target.shape)
        # print("pred_1: ", pred.shape)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        # print("mask: ", mask.shape)

        # print("loss_shape_before: ", loss.shape)
        loss = loss[:, [0, 1, 2], :].mean(dim=1) # (N,L)
        # print("loss_shape_after: ", loss.shape)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        imgs_wo_dem = imgs[:, :-1, : ,:] # pass all channels except DEM (last channel)
        dem_orig = imgs[:, -1, : ,:] # pass only DEM
        dem = dem_orig.unsqueeze(1)

        latent, mask, ids_restore = self.forward_encoder(imgs_wo_dem, mask_ratio)

        latent_dem, mask_dem, ids_restore_dem = self.forward_encoder_dem(dem, mask_ratio)
        
        # saugat: gating and element-wise multiplication (GLU operation) after each encoder block
        # latent_dem = self.dem_sigmoid(latent_dem) # gating

        # latent = latent * latent_dem # fusing
        latent = latent + latent_dem # fusing

        # append cls token
        cls_tokens = self.cls_token.expand(latent.shape[0], -1, -1)
        latent = torch.cat((cls_tokens, latent), dim=1)  # (N, L + 1, D)

        # apply transformer blocks
        for blk in self.blocks:
            latent = blk(latent)
        
        latent = self.norm(latent)

        latent_dec = self.forward_decoder(latent, dem, ids_restore)  # [N, L, p*p*3] # dem is original dem (unmasked)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            latent_dec = blk(latent_dec)
        latent_dec = self.decoder_norm(latent_dec)

        # predictor projection
        pred = self.decoder_pred(latent_dec)

        # remove cls token
        pred = pred[:, 1:, :]

        loss = self.forward_loss(imgs_wo_dem, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
