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

		self.segmentation_head = SegmentationHead(self.embed_dim, num_classes)

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
		x = self.segmentation_head(x)
		return x
