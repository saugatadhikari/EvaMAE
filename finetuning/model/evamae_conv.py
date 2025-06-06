from model.segmentation_head import SegmentationHead
import torch
from timm.models.vision_transformer import VisionTransformer, PatchEmbed, Block
import numpy as np
from torch import nn


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
	"""
	grid_size: int of the grid height and width
	return:
	pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
	"""
	grid_h = np.arange(grid_size, dtype=np.float32)
	grid_w = np.arange(grid_size, dtype=np.float32)
	grid = np.meshgrid(grid_w, grid_h)  # here w goes first
	grid = np.stack(grid, axis=0)

	grid = grid.reshape([2, 1, grid_size, grid_size])
	pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
	if cls_token:
		pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
	return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
	assert embed_dim % 2 == 0

	# use half of dimensions to encode grid_h
	emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
	emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

	emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
	return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
	"""
	embed_dim: output dimension for each position
	pos: a list of positions to be encoded: size (M,)
	out: (M, D)
	"""
	assert embed_dim % 2 == 0
	omega = np.arange(embed_dim // 2, dtype=np.float32)
	omega /= embed_dim / 2.
	omega = 1. / 10000 ** omega  # (D/2,)

	pos = pos.reshape(-1)  # (M,)
	out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

	emb_sin = np.sin(out)  # (M, D/2)
	emb_cos = np.cos(out)  # (M, D/2)

	emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
	return emb

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


class EvaMAE_Conv(VisionTransformer):
	def __init__(self, num_classes=1,
			  		img_size=224,
					in_chans=3,
					patch_size=16,
					embed_dim=1024,
					depth=24,
					num_heads=16,
					mlp_ratio=4,
					qkv_bias=True,
					norm_layer=nn.LayerNorm, 
                    conv_kernel_size=3,
					**kwargs):
		
		super(EvaMAE_Conv, self).__init__(**kwargs)

		self.embed_dim = embed_dim
		self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
		num_patches = self.patch_embed.num_patches
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
		self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

		self.patch_embed_dem = PatchEmbed(img_size, patch_size, 1, embed_dim)

		self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
		self.norm = norm_layer(embed_dim)

		# Conv layer specifics
		self.inc = inconv(1, 1, conv_kernel_size)

		pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
											cls_token=True)
		self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
		del self.head

		self.dem_sigmoid = torch.nn.Sigmoid()

		self.segmentation_head = SegmentationHead(self.embed_dim, num_classes)

	def forward_features(self, x):
		B = x.shape[0]
		x = self.patch_embed(x)
		cls_tokens = self.cls_token.expand(B, -1, -1)
		x = torch.cat((cls_tokens, x), dim=1)
		x = x + self.pos_embed
		x = self.pos_drop(x)
		return x
    
	def forward_features_dem(self, x):
		b, c, h, w = x.shape
            
		x = self.inc(x)
		
		x = self.patch_embed_dem(x)
            
		_, L, D = x.shape
            
		cls_tokens = self.cls_token.expand(b, -1, -1)
		x = torch.cat((cls_tokens, x), dim=1)
		x = x + self.pos_embed
		x = self.pos_drop(x)
		return x

	def forward(self, x):
		imgs_wo_dem = x[:, :-1, : ,:] # pass all channels except DEM (last channel)
		dem_orig = x[:, -1, : ,:] # pass only DEM
		dem = dem_orig.unsqueeze(1)

		latent = self.forward_features(imgs_wo_dem.to(self.device)) # pretrained encoder
		latent_dem = self.forward_features_dem(dem.to(self.device))
            
		# saugat: gating and element-wise addition (GLU operation) after each encoder block
		latent_dem = self.dem_sigmoid(latent_dem) # gating
		latent = latent + latent_dem # fusing

		# apply transformer blocks
		for blk in self.blocks:
			latent = blk(latent)

		latent = self.norm(latent)
            
		latent = latent[:, :-1, :] # remove cls token

		B, N, C = latent.shape # batch_size, (num_patches * num_patches), embed_dim (1024)
		latent = latent.transpose(1, 2).view(B, C, int(N ** 0.5), int(N ** 0.5))  # reshape to (B, C, H, W) -> (4, 1024, 14, 14)
		latent = self.segmentation_head(latent)
		return latent
