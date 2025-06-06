import torch
import torch.nn as nn
import torch.nn.functional as F
from model.segmentation_head import SegmentationHead
from timm.models.vision_transformer import VisionTransformer, PatchEmbed, Block
import numpy as np

from copy import deepcopy


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

class ZeroConv1x1(nn.Module):
    """Zero-initialized 1x1 convolution."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)

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
		
		return latent

class EvaMAE_Channel(VisionTransformer):
	def __init__(self, num_classes=1,
			  		img_size=224,
					in_chans=4,
					patch_size=16,
					embed_dim=1024,
					depth=24,
					num_heads=16,
					mlp_ratio=4,
					qkv_bias=True,
					norm_layer=nn.LayerNorm, 
					**kwargs):
		
		super(EvaMAE_Channel, self).__init__(**kwargs)

		self.embed_dim = embed_dim
		self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
		num_patches = self.patch_embed.num_patches

		self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
		self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

		self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
		self.norm = norm_layer(embed_dim)

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	
		pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
											cls_token=True)
		self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
		del self.head

	def forward_features(self, x):
		B = x.shape[0]
		x = self.patch_embed(x)
		cls_tokens = self.cls_token.expand(B, -1, -1)
		x = torch.cat((cls_tokens, x), dim=1)
		x = x + self.pos_embed	
		x = self.pos_drop(x)
		for blk in self.blocks:
			x = blk(x)

		return x[:, 1:, :]  # return all tokens except cls token

	def forward(self, x):
		x = self.forward_features(x.to(self.device)) # pretrained encoder
		B, N, C = x.shape # batch_size, (num_patches * num_patches), embed_dim (1024)
		x = x.transpose(1, 2).view(B, C, int(N ** 0.5), int(N ** 0.5))  # reshape to (B, C, H, W) -> (4, 1024, 14, 14)
		return x

# Cross-attention layer where RGB tokens attend to DEM tokens
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, q, k, v):
        return self.attn(q, k, v)[0]  # Cross-attention # Each RGB token looks at all DEM tokens to decide which DEM features are most useful for reconstruction or classification.

class EvaMAE_CrossViT(VisionTransformer):
	def __init__(self, num_classes=3,
			  		img_size=224,
					in_chans=3,
					patch_size=16,
					embed_dim=1024,
					depth=24,
					num_heads=16,
					mlp_ratio=4,
					qkv_bias=True,
					norm_layer=nn.LayerNorm,
					use_evanet=False,
					**kwargs):
		
		super(EvaMAE_CrossViT, self).__init__(**kwargs)

		self.embed_dim = embed_dim
		self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
		num_patches = self.patch_embed.num_patches
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.use_evanet = use_evanet

		self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
		self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

		self.patch_embed_dem = PatchEmbed(img_size, patch_size, 1, embed_dim)

		self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
		self.norm = norm_layer(embed_dim)

		pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
											cls_token=True)
		self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
		del self.head

		self.cross_attn_rgb_to_dem = CrossAttention(embed_dim, num_heads=8)

		self.norm_cross = nn.LayerNorm(embed_dim)
		self.cross_mlp = nn.Sequential(
			nn.Linear(embed_dim, embed_dim * mlp_ratio),
			nn.GELU(),
			nn.Linear(embed_dim * mlp_ratio, embed_dim),
		)
		self.norm_cross2 = nn.LayerNorm(embed_dim)

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

		latent_dem_encoded = torch.clone(latent_dem)

		# apply transformer blocks
		block_count = 0
		for blk in self.blocks:
			latent = blk(latent)

			if block_count == 0:
				latent_dem = blk(latent_dem) # only pass through 1 layer

			block_count += 1

		# Apply Cross-Attention: RGB queries DEM
		latent_cross_out = self.cross_attn_rgb_to_dem(latent, latent_dem, latent_dem)
		latent = latent + latent_cross_out
		latent = self.norm_cross(latent)
		latent = latent + self.cross_mlp(latent)
		latent = self.norm_cross2(latent)
		
		latent = latent[:, :-1, :] # remove cls token
		latent_dem_encoded = latent_dem_encoded[:, :-1, :] # remove cls token

		B, N, C = latent.shape # batch_size, (num_patches * num_patches), embed_dim (1024)
		latent = latent.transpose(1, 2).view(B, C, int(N ** 0.5), int(N ** 0.5))  # reshape to (B, D, H, W) -> (4, 1024, 14, 14)
		latent_dem_encoded = latent_dem_encoded.transpose(1, 2).view(B, C, int(N ** 0.5), int(N ** 0.5))  # reshape to (B, D, H, W) -> (4, 1024, 14, 14)
		
		return latent

class EvaMAE_CrossAttn(VisionTransformer):
	def __init__(self, num_classes=3,
			  		img_size=224,
					in_chans=3,
					patch_size=16,
					embed_dim=1024,
					depth=24,
					num_heads=16,
					mlp_ratio=4,
					qkv_bias=True,
					norm_layer=nn.LayerNorm,
					use_evanet=False,
					**kwargs):
		
		super(EvaMAE_CrossAttn, self).__init__(**kwargs)

		self.embed_dim = embed_dim
		self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
		num_patches = self.patch_embed.num_patches
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.use_evanet = use_evanet

		self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
		self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

		self.patch_embed_dem = PatchEmbed(img_size, patch_size, 1, embed_dim)

		self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
		self.norm = norm_layer(embed_dim)

		pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
											cls_token=True)
		self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
		del self.head

		self.cross_attn_rgb_to_dem = CrossAttention(embed_dim, num_heads=8)

		self.norm_cross = nn.LayerNorm(embed_dim)
		self.cross_mlp = nn.Sequential(
			nn.Linear(embed_dim, embed_dim * mlp_ratio),
			nn.GELU(),
			nn.Linear(embed_dim * mlp_ratio, embed_dim),
		)
		self.norm_cross2 = nn.LayerNorm(embed_dim)

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

		latent_dem_encoded = torch.clone(latent_dem)

		# apply transformer blocks
		block_count = 0
		for blk in self.blocks:
			latent = blk(latent)

			if block_count == 0:
				latent_dem = blk(latent_dem) # only pass through 1 layer

			block_count += 1

		# Apply Cross-Attention: RGB queries DEM
		latent = self.cross_attn_rgb_to_dem(latent, latent_dem, latent_dem)
		latent = self.norm_cross(latent)
            
		latent = latent[:, :-1, :] # remove cls token
		latent_dem_encoded = latent_dem_encoded[:, :-1, :] # remove cls token

		B, N, C = latent.shape # batch_size, (num_patches * num_patches), embed_dim (1024)
		latent = latent.transpose(1, 2).view(B, C, int(N ** 0.5), int(N ** 0.5))  # reshape to (B, D, H, W) -> (4, 1024, 14, 14)
		latent_dem_encoded = latent_dem_encoded.transpose(1, 2).view(B, C, int(N ** 0.5), int(N ** 0.5))  # reshape to (B, D, H, W) -> (4, 1024, 14, 14)
		
		return latent


# ---- Main Model with DEM Conditioning ---- #
class ControlNet(nn.Module):
    def __init__(self, num_classes, embed_dim, model, pretrained_weight_path, use_dem, inference=False):
        super().__init__()
        self.use_dem = use_dem
        self.embed_dim = embed_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not inference:
            pretrained_model = torch.load(pretrained_weight_path, map_location='cpu')['model']

        if model == "evamae_channel":
            pretrained_encoder = EvaMAE_Channel(num_classes=num_classes,
						embed_dim=1024,
						depth=24,
						num_heads=16,
						mlp_ratio=4,
						qkv_bias=True)
        elif model == "evamae_crossvit":
            pretrained_encoder = EvaMAE_CrossViT(num_classes=num_classes,
						embed_dim=1024,
						depth=24,
						num_heads=16,
						mlp_ratio=4,
						qkv_bias=True)
        elif model == "evamae_crossattn":
            pretrained_encoder = EvaMAE_CrossAttn(num_classes=num_classes,
						embed_dim=1024,
						depth=24,
						num_heads=16,
						mlp_ratio=4,
						qkv_bias=True)
        elif model == 'evamae_conv':
            pretrained_encoder = EvaMAE_Conv(num_classes=num_classes,
						img_size=224,
						in_chans=3,
						patch_size=16,
					   embed_dim=1024,
					   depth=24,
					   num_heads=16,
					   mlp_ratio=4,
					   qkv_bias=True,
					   conv_kernel_size=3)

        if not inference:
            model_dict = pretrained_encoder.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_model.items() if k in model_dict and 'head' not in k}
            model_dict.update(pretrained_dict)
            pretrained_encoder.load_state_dict(model_dict)

		
        self.pretrained_encoder = deepcopy(pretrained_encoder)  # frozen version
        self.pretrained_encoder.requires_grad_(False)
        
        self.trainable_encoder_copy = deepcopy(pretrained_encoder) # trainable version
        
        self.segmentation_head = SegmentationHead(self.embed_dim, num_classes)

        self.zero_conv_in = ZeroConv1x1(1, 3)
        self.zero_conv_out = ZeroConv1x1(1024, 1024)

    def forward(self, x):
        rgb = x[:, :-1, : ,:] # pass all channels except DEM (last channel)
        dem_orig = x[:, -1, : ,:] # pass only DEM
        dem = dem_orig.unsqueeze(1)
        rgb = rgb.to(self.device)
        dem = dem.to(self.device)

		# Frozen
        with torch.no_grad():
            if self.use_dem:
                rgb = torch.cat([rgb, dem], dim=1)
            rgb_feats = self.pretrained_encoder(rgb)  # frozen features -> (4, 1024, 14, 14)

		# for DEM conditioning using ControlNet
        rgb[:,:3,:,:] = rgb[:,:3,:,:] + self.zero_conv_in(dem)
        injected = self.trainable_encoder_copy(rgb)

        conditioned_feat = rgb_feats + self.zero_conv_out(injected)

        out = self.segmentation_head(conditioned_feat)
        return out
