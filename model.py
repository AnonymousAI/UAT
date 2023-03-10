# coding=utf-8
import math
import random
import functools
import operator
import numpy as np
from ipdb import set_trace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import tensor_transforms as tt
from spectral import SpectralNorm
from miscc.config import cfg
from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
from distributed import get_rank
from tools.blocks import ConstantInput, StyledConv, ToRGB, PixelNorm, EqualLinear, Unfold, LFF
class GLU(nn.Module):
	def __init__(self):
		super(GLU, self).__init__()

	def forward(self, x):  # (N,c_dim*4)
		nc = x.size(1)  # c_dim*4
		assert nc % 2 == 0, 'channels dont divide 2!'
		nc = int(nc/2)  # c_dim*2
		return x[:, :nc] * torch.sigmoid(x[:, nc:])  # c_dim*2


# ############## G networks ###################
class CA_NET(nn.Module):
	# some code is modified from vae examples
	# (https://github.com/pytorch/examples/blob/master/vae/main.py)
	def __init__(self, in_dim, out_dim):
		super(CA_NET, self).__init__()
		self.t_dim = in_dim   # 256
		self.c_dim = out_dim  # 100
		self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
		self.relu = GLU()

	def encode(self, text_embedding):
		x = self.relu(self.fc(text_embedding))  # (N,c_dim*2)
		mu = x[:, :self.c_dim]      # (N,c_dim)
		logvar = x[:, self.c_dim:]  # (N,c_dim)
		return mu, logvar

	def reparametrize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()  # (N,c_dim)
		if cfg.CUDA:
			eps = torch.cuda.FloatTensor(std.size()).normal_()  # (N,c_dim)
		else:
			eps = torch.FloatTensor(std.size()).normal_()
		eps = Variable(eps)
		return eps.mul(std).add_(mu)  # (N,c_dim)

	def forward(self, text_embedding):
		mu, logvar = self.encode(text_embedding)  # (N,c_dim), (N,c_dim)
		c_code = self.reparametrize(mu, logvar)
		return c_code, mu, logvar  # (N,c_dim), (N,c_dim), (N,c_dim)


class PixelNorm(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
	k = torch.tensor(k, dtype=torch.float32)

	if k.ndim == 1:
		k = k[None, :] * k[:, None]

	k /= k.sum()

	return k


class Upsample(nn.Module):
	def __init__(self, kernel, factor=2):
		super().__init__()

		self.factor = factor
		kernel = make_kernel(kernel) * (factor ** 2)
		self.register_buffer('kernel', kernel)

		p = kernel.shape[0] - factor

		pad0 = (p + 1) // 2 + factor - 1
		pad1 = p // 2

		self.pad = (pad0, pad1)

	def forward(self, input):
		out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

		return out


class Downsample(nn.Module):
	def __init__(self, kernel, factor=2):
		super().__init__()

		self.factor = factor
		kernel = make_kernel(kernel)
		self.register_buffer('kernel', kernel)

		p = kernel.shape[0] - factor

		pad0 = (p + 1) // 2
		pad1 = p // 2

		self.pad = (pad0, pad1)

	def forward(self, input):
		out = upfirdn2d(input, self.kernel, down=self.factor, pad=self.pad)

		return out


class Blur(nn.Module):
	def __init__(self, kernel, pad, upsample_factor=1):
		super().__init__()

		kernel = make_kernel(kernel)

		if upsample_factor > 1:
			kernel = kernel * (upsample_factor ** 2)

		self.register_buffer('kernel', kernel)

		self.pad = pad

	def forward(self, input):
		out = upfirdn2d(input, self.kernel, pad=self.pad)

		return out


class EqualConv2d(nn.Module):
	def __init__(
		self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
	):
		super().__init__()

		self.weight = nn.Parameter(
			torch.randn(out_channel, in_channel, kernel_size, kernel_size)
		)
		self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

		self.stride = stride
		self.padding = padding

		if bias:
			self.bias = nn.Parameter(torch.zeros(out_channel))

		else:
			self.bias = None

	def forward(self, input):
		out = F.conv2d(
			input,
			self.weight * self.scale,
			bias=self.bias,
			stride=self.stride,
			padding=self.padding,
		)

		return out

	def __repr__(self):
		return (
			f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
			f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
		)

		
class EqualLinear(nn.Module):
	def __init__(
		self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
	):
		super().__init__()

		self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

		if bias:
			self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

		else:
			self.bias = None

		self.activation = activation

		self.scale = (1 / math.sqrt(in_dim)) * lr_mul
		self.lr_mul = lr_mul

	def forward(self, input):

		if self.activation:
			out = F.linear(input, self.weight * self.scale)
			out = fused_leaky_relu(out, self.bias * self.lr_mul)

		else:
			out = F.linear(
				input, self.weight * self.scale, bias=self.bias * self.lr_mul
			)

		return out

	def __repr__(self):
		return (
			f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
		)


class ModulatedConv2d(nn.Module):
	def __init__(
		self,
		in_channel,
		out_channel,
		kernel_size,
		style_dim,
		demodulate=True,
		upsample=False,
		downsample=False,
		blur_kernel=[1, 3, 3, 1],
	):
		super().__init__()

		self.eps = 1e-8
		self.kernel_size = kernel_size
		self.in_channel = in_channel
		self.out_channel = out_channel
		self.upsample = upsample
		self.downsample = downsample

		if upsample:
			factor = 2
			p = (len(blur_kernel) - factor) - (kernel_size - 1)
			pad0 = (p + 1) // 2 + factor - 1
			pad1 = p // 2 + 1

			self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

		if downsample:
			factor = 2
			p = (len(blur_kernel) - factor) + (kernel_size - 1)
			pad0 = (p + 1) // 2
			pad1 = p // 2

			self.blur = Blur(blur_kernel, pad=(pad0, pad1))

		fan_in = in_channel * kernel_size ** 2
		self.scale = 1 / math.sqrt(fan_in)
		self.padding = kernel_size // 2

		self.weight = nn.Parameter(
			torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
		)
		
		# self.weight = nn.Parameter(
		# 	torch.randn(out_channel, in_channel, kernel_size, kernel_size)
		# )

		self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

		self.demodulate = demodulate

	def __repr__(self):
		return (
			f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
			f'upsample={self.upsample}, downsample={self.downsample})'
		)

	def forward(self, input, style):
		batch, in_channel, height, width = input.shape

		style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
		weight = self.scale * self.weight * style
		# weight = self.scale * self.weight

		if self.demodulate:
			demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
			weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

		weight = weight.view(
			batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
		)

		if self.upsample:
			input = input.view(1, batch * in_channel, height, width)
			weight = weight.view(
				batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
			)
			weight = weight.transpose(1, 2).reshape(
				batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
			)
			out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
			_, _, height, width = out.shape
			out = out.view(batch, self.out_channel, height, width)
			out = self.blur(out)
			
			# weight = weight.transpose(0, 1)
			# out = self.blur(F.conv_transpose2d(input, weight, padding=0, stride=2))

		elif self.downsample:
			input = self.blur(input)
			_, _, height, width = input.shape
			input = input.view(1, batch * in_channel, height, width)
			out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
			_, _, height, width = out.shape
			out = out.view(batch, self.out_channel, height, width)
			
			# out = F.conv2d(self.blur(input), weight, padding=0, stride=2)

		else:
			input = input.view(1, batch * in_channel, height, width)
			out = F.conv2d(input, weight, padding=self.padding, groups=batch)
			_, _, height, width = out.shape
			out = out.view(batch, self.out_channel, height, width)

			# out = F.conv2d(input, weight, padding=self.padding)

		return out


class NoiseInjection(nn.Module):
	def __init__(self):
		super().__init__()

		self.weight = nn.Parameter(torch.zeros(1))

	def forward(self, image, noise=None):
		if noise is None:
			batch, _, height, width = image.shape
			noise = image.new_empty(batch, 1, height, width).normal_()

		return image + self.weight * noise


class ConstantInput(nn.Module):
	def __init__(self, channel, size=4):
		super().__init__()

		self.input = nn.Parameter(torch.randn(1, channel, size, size))

	def forward(self, x):
		batch = x.shape[0]
		out = self.input.repeat(batch, 1, 1, 1)

		return out


class StyledConv(nn.Module):
	def __init__(
		self,
		in_channel,
		out_channel,
		kernel_size,
		style_dim,
		upsample=False,
		blur_kernel=[1, 3, 3, 1],
		demodulate=True,
	):
		super().__init__()

		self.conv = ModulatedConv2d(
			in_channel,
			out_channel,
			kernel_size,
			style_dim,
			upsample=upsample,
			blur_kernel=blur_kernel,
			demodulate=demodulate,
		)

		self.noise = NoiseInjection()
		# self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
		# self.activate = ScaledLeakyReLU(0.2)
		self.activate = FusedLeakyReLU(out_channel)
		
		# self.norm = nn.InstanceNorm2d(out_channel)
		
		# self.modulation = EqualLinear(style_dim, out_channel, bias_init=1)

	def forward(self, input, style, noise=None):
		out = self.conv(input, style)
		out = self.noise(out, noise=noise)
		# out = out + self.bias
		out = self.activate(out)

		# # InstanceNorm
		# out = self.norm(out)

		# # Style_Mod
		# batch, dim, h, w = out.shape
		# style = self.modulation(style).view(batch, dim, 1, 1)
		# # out = out * (style + 1)
		# out = out * style

		return out


class ToRGB(nn.Module):
	def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
		super().__init__()

		if upsample:
			self.upsample = Upsample(blur_kernel)

		self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
		self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))
		# self.tanh = nn.Tanh()

	def forward(self, input, style, skip=None):
		out = self.conv(input, style)
		out = out + self.bias

		if skip is not None:
			skip = self.upsample(skip)
			out = out + skip
		
		# out = self.tanh(out)
		
		return out


class Generator(nn.Module):
	def __init__(
		self, 
		size, 
		channel_multiplier=2, 
		blur_kernel=[1, 3, 3, 1], 
		lr_mlp=0.01
	):
		
		super().__init__()

		self.size = size
		
		self.nef = cfg.TEXT.EMBEDDING_DIM

		self.w_dim = cfg.GAN.W_DIM	
		
		self.n_mlp = cfg.GAN.N_MLP

		# layers = [PixelNorm()]
		self.pixel_norm = PixelNorm()

		layers = []
		for i in range(self.n_mlp):
			layers.append(
				EqualLinear(
					self.w_dim, self.w_dim, lr_mul=lr_mlp, activation='fused_lrelu'
				)
			)

		# self.style = nn.Sequential(*layers)
		self.mapping = nn.Sequential(*layers)

		self.channels = {
			4: 512,
			8: 512,
			16: 512,
			32: 512,
			64: 256 * channel_multiplier,
			128: 128 * channel_multiplier,
			256: 64 * channel_multiplier,
			512: 32 * channel_multiplier,
			1024: 16 * channel_multiplier,
		}

		# self.fc1 = EqualLinear(
		# 	self.nef, self.w_dim, lr_mul=lr_mlp, activation='fused_lrelu'
		# )

		# self.ca_net = CA_NET(self.nef, cfg.GAN.C_DIM)
		# in_dim = cfg.GAN.C_DIM + cfg.GAN.Z_DIM
		# out_dim = self.channels[4] * 4 * 4
		# self.fc = nn.Sequential(
		# 	nn.Linear(in_dim, out_dim * 2, bias=False),
		# 	nn.BatchNorm1d(out_dim * 2),
		# 	GLU()
		# )

		self.const_input = ConstantInput(self.channels[4])
		self.conv1 = StyledConv(
			self.channels[4], self.channels[4], 3, self.w_dim, blur_kernel=blur_kernel
		)
		self.to_rgb1 = ToRGB(self.channels[4], self.w_dim, upsample=False)

		self.log_size = int(math.log(size, 2))
		self.num_layers = (self.log_size - 2) * 2 + 1

		self.convs = nn.ModuleList()
		self.upsamples = nn.ModuleList()
		self.to_rgbs = nn.ModuleList()
		self.noises = nn.Module()

		in_channel = self.channels[4]

		for layer_idx in range(self.num_layers):
			res = (layer_idx + 5) // 2
			shape = [1, 1, 2 ** res, 2 ** res]
			self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

		for i in range(3, self.log_size + 1):
			out_channel = self.channels[2 ** i]

			self.convs.append(
				StyledConv(
					in_channel,
					out_channel,
					3,
					self.w_dim,
					upsample=True,
					blur_kernel=blur_kernel,
				)
			)

			self.convs.append(
				StyledConv(
					out_channel, out_channel, 3, self.w_dim, blur_kernel=blur_kernel
				)
			)

			self.to_rgbs.append(ToRGB(out_channel, self.w_dim))

			in_channel = out_channel

		self.n_latent = self.log_size * 2 - 2

	def make_noise(self):
		device = self.const_input.input.device

		noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

		for i in range(3, self.log_size + 1):
			for _ in range(2):
				noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

		return noises

	def mean_latent(self, n_latent):
		latent_in = torch.randn(
			n_latent, self.w_dim, device=self.const_input.input.device
		)
		latent = self.mapping(latent_in).mean(0, keepdim=True)

		return latent

	def get_latent(self, x):
		return self.mapping(x)

	def forward(
		self,
		sents, 
		return_latents=False,
		inject_index=None,
		truncation=1,
		truncation_latent=None,
		input_is_latent=False,
		noise=None,
		randomize_noise=True,
	):	

		mu, logvar = None, None
		# c_code, mu, logvar = self.ca_net(sents)  # N,100
		# c_z_code = torch.cat((c_code, z_code), 1)

		# Latent proj
		# sents = self.fc1(sents)  # N,512
		# print('sents: [%.4f, %.4f]' % (sents.min(), sents.max()))

		# Normalize
		# sents = F.normalize(sents, p=2, dim=1, eps=1e-8)
		sents = self.pixel_norm(sents)
		# print('styles_norm: [%.4f, %.4f]' % (sents.min(), sents.max()))

		# Mapping
		if not input_is_latent:
			dlatents = self.mapping(sents)
			# print('dlatents: [%.4f, %.4f]' % (dlatents.min(), dlatents.max()))
		
		# TODO ?
		# Update moving average of W. 

		# Apply truncation trick.
		if truncation < 1:
			dlatents = truncation_latent + truncation * (dlatents - truncation_latent)

		# Broadcast
		inject_index = self.n_latent
		if dlatents.ndim < 3:
			dlatents = dlatents.unsqueeze(1).repeat(1, inject_index, 1)
		else:
			dlatents = dlatents

		# Noise
		if noise is None:
			if randomize_noise:
				noise = [None] * self.num_layers
			else:
				noise = [
					getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
				]
		
		out = self.const_input(dlatents)
		# out = self.fc(c_z_code).view(-1, self.channels[4], 4, 4)  # (N,512,4,4)

		out = self.conv1(out, dlatents[:, 0], noise=noise[0])

		skip = self.to_rgb1(out, dlatents[:, 1])

		i = 1
		for conv1, conv2, noise1, noise2, to_rgb in zip(
			self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
		):
			out = conv1(out, dlatents[:, i], noise=noise1)
			out = conv2(out, dlatents[:, i + 1], noise=noise2)
			skip = to_rgb(out, dlatents[:, i + 2], skip)

			i += 2

		image = skip

		if return_latents:
			return image, mu, logvar, dlatents

		else:
			return image, mu, logvar, None

class ScaledLeakyReLU(nn.Module):
	def __init__(self, negative_slope=0.2):
		super().__init__()

		self.negative_slope = negative_slope

	def forward(self, input):
		out = F.leaky_relu(input, negative_slope=self.negative_slope)

		return out * math.sqrt(2)


class ConvLayer(nn.Sequential):
	def __init__(
		self,
		in_channel,
		out_channel,
		kernel_size,
		downsample=False,
		blur_kernel=[1, 3, 3, 1],
		bias=True,
		activate=True,
	):
		layers = []

		if downsample:
			factor = 2
			p = (len(blur_kernel) - factor) + (kernel_size - 1)
			pad0 = (p + 1) // 2
			pad1 = p // 2

			layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

			stride = 2
			self.padding = 0

		else:
			stride = 1
			self.padding = kernel_size // 2

		layers.append(
			EqualConv2d(
				in_channel,
				out_channel,
				kernel_size,
				padding=self.padding,
				stride=stride,
				bias=bias and not activate,
			)
		)

		if activate:
			if bias:
				layers.append(FusedLeakyReLU(out_channel))

			else:
				layers.append(ScaledLeakyReLU(0.2))

		super().__init__(*layers)


class ResBlock(nn.Module):
	def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
		super().__init__()

		self.conv1 = ConvLayer(in_channel, in_channel, 3)
		self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

		self.skip = ConvLayer(
			in_channel, out_channel, 1, downsample=True, activate=False, bias=False
		)

	def forward(self, input):
		out = self.conv1(input)
		out = self.conv2(out)

		skip = self.skip(input)
		out = (out + skip) / math.sqrt(2)

		return out


class Discriminator(nn.Module):
	def __init__(
		self,
		size,
		channel_multiplier=2, 
		blur_kernel=[1, 3, 3, 1]
	):
		super().__init__()

		channels = {
			4: 512,
			8: 512,
			16: 512,
			32: 512,
			64: 256 * channel_multiplier,
			128: 128 * channel_multiplier,
			256: 64 * channel_multiplier,
			512: 32 * channel_multiplier,
			1024: 16 * channel_multiplier,
		}

		convs = [ConvLayer(3, channels[size], 1)]

		log_size = int(math.log(size, 2))

		in_channel = channels[size]

		for i in range(log_size, 2, -1):
			out_channel = channels[2 ** (i - 1)]

			convs.append(ResBlock(in_channel, out_channel, blur_kernel))

			in_channel = out_channel

		self.convs = nn.Sequential(*convs)

		self.stddev_group = 4
		self.stddev_feat = 1

		self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
		self.final_linear = nn.Sequential(
			EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
			EqualLinear(channels[4], 1),
		)
		nef = cfg.TEXT.EMBEDDING_DIM
		
		self.COND_DNET = D_GET_LOGITS(channels[4], nef, bcondition=True)

	def forward(self, image, c_code=None):
		out = self.convs(image)   

		batch, channel, height, width = out.shape  # N,512,4,4
		group = min(batch, self.stddev_group)
		stddev = out.view(
			group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
		)
		stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
		stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
		stddev = stddev.repeat(group, 1, height, width)
		out = torch.cat([out, stddev], 1)

		out = self.final_conv(out)
		if c_code is not None:
			cond_logits = self.COND_DNET(out, c_code)
		else:
			cond_logits = None
			
		out = out.view(batch, -1)
		out = self.final_linear(out)
		
		return out, cond_logits



# ############## D networks ##########################
def conv3x3(in_planes, out_planes, bias=False):
	"3x3 convolution with padding"
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=bias)


def Block3x3_leakRelu(in_planes, out_planes):
	block = nn.Sequential(
		SpectralNorm(conv3x3(in_planes, out_planes, bias=True)),
		nn.LeakyReLU(0.2, inplace=True)
	)
	return block


# Downscale the spatial size by a factor of 2
def downBlock(in_planes, out_planes):
	block = nn.Sequential(
		SpectralNorm(nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=True)),
		nn.LeakyReLU(0.2, inplace=True)
	)
	return block


# Downscale the spatial size by a factor of 16
def encode_image_by_16times(ndf):
	layers = []
	layers.append(SpectralNorm(nn.Conv2d(3, ndf, 4, 2, 1, bias=True)))
	layers.append(nn.LeakyReLU(0.2, inplace=True),)
	layers.append(SpectralNorm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True)))
	layers.append(nn.LeakyReLU(0.2, inplace=True))
	layers.append(SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True)))
	layers.append(nn.LeakyReLU(0.2, inplace=True))
	layers.append(SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True)))
	layers.append(nn.LeakyReLU(0.2, inplace=True))
	return nn.Sequential(*layers)


class D_GET_LOGITS(nn.Module):
	def __init__(self, ndf, nef, bcondition=False):
		super(D_GET_LOGITS, self).__init__()
		self.df_dim = ndf
		self.ef_dim = nef
		self.bcondition = bcondition
		if self.bcondition:
			self.jointConv = Block3x3_leakRelu(ndf + nef, ndf)

		self.outlogits = nn.Sequential(
			nn.Conv2d(ndf, 1, kernel_size=4, stride=4),
			nn.Sigmoid())

	def forward(self, h_code, c_code=None):
		if self.bcondition and c_code is not None:
			# conditioning output
			c_code = c_code.view(-1, self.ef_dim, 1, 1)
			c_code = c_code.repeat(1, 1, 4, 4)
			# state size (ngf+egf) x 4 x 4
			h_c_code = torch.cat((h_code, c_code), 1)
			# state size ngf x in_size x in_size
			h_c_code = self.jointConv(h_c_code)
		else:
			h_c_code = h_code

		output = self.outlogits(h_c_code)
		return output.view(-1)


class D_GET_LOGITS_trainer22(nn.Module):
	def __init__(self, ndf, nef, bcondition=False):
		super(D_GET_LOGITS_trainer22, self).__init__()
		self.df_dim = ndf
		self.ef_dim = nef
		self.bcondition = bcondition
		if self.bcondition:
			self.jointConv = Block3x3_leakRelu(ndf + nef, ndf)

		# self.outlogits = nn.Sequential(
		# 	nn.Conv2d(ndf, 1, kernel_size=4, stride=4))

	def forward(self, h_code, c_code=None):
		if self.bcondition and c_code is not None:
			# conditioning output
			c_code = c_code.view(-1, self.ef_dim, 1, 1)
			c_code = c_code.repeat(1, 1, 4, 4)
			# state size (ngf+egf) x 4 x 4
			h_c_code = torch.cat((h_code, c_code), 1)
			# state size ngf x in_size x in_size
			h_c_code = self.jointConv(h_c_code)
		else:
			h_c_code = h_code
		#print('1:',h_c_code.shape)
		# output = self.outlogits(h_c_code)
		# print('2:',output.shape)
		return h_c_code


# For 256 x 256 images
class D_NET256(nn.Module):
	def __init__(self, b_jcu=True):
		super(D_NET256, self).__init__()
		ndf = cfg.GAN.DF_DIM
		nef = cfg.TEXT.EMBEDDING_DIM
		self.img_code_s16 = encode_image_by_16times(ndf)
		self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
		self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
		self.img_code_s64_1 = Block3x3_leakRelu(ndf * 32, ndf * 16)
		self.img_code_s64_2 = Block3x3_leakRelu(ndf * 16, ndf * 8)
		if b_jcu:
			self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
		else:
			self.UNCOND_DNET = None
		self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)

	def forward(self, x_var):
		x_code16 = self.img_code_s16(x_var)
		x_code8 = self.img_code_s32(x_code16)
		x_code4 = self.img_code_s64(x_code8)
		x_code4 = self.img_code_s64_1(x_code4)
		x_code4 = self.img_code_s64_2(x_code4)
		return x_code4


