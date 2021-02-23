"""
author: Antoine Spahr

date : 05.11.2020

----------

TO DO :
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dLayer(nn.Module):
    """
    Define a 2D Convolution Layer with possibility to add spectral Normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True,
                 padding_mode='zeros', activation='relu', batch_norm=True, sn=False, power_iter=1):
        """
        Build a Gated 2D convolutional layer as a pytorch nn.Module.
        ----------
        INPUT
            |---- in_channels (int) the number of input channels of the convolutions.
            |---- out_channels (int) the number of output channels of the convolutions.
            |---- kernel_size (int) the kernel size of the convolutions.
            |---- stride (int) the stride of the convolution kernel.
            |---- padding (int) the padding of the input prior to the convolution.
            |---- dilation (int) the dilation of the kernel.
            |---- bias (bool) wether to use a bias term on the convolution kernel.
            |---- padding_mode (str) how to pad the image (see nn.Conv2d doc for details).
            |---- activation (str) the activation function to use. Supported: 'relu' -> ReLU, 'lrelu' -> LeakyReLU,
            |               'prelu' -> PReLU, 'selu' -> SELU, 'tanh' -> Hyperbolic tangent, 'sigmoid' -> sigmoid,
            |               'none' -> No activation used
            |---- batch_norm (bool) whether to use a batch normalization layer between the convolution and the activation.
            |---- sn (bool) whether to use Spetral Normalization on the convolutional weights.
            |---- power_iter (int) the number of iteration for Spectral norm estimation.
        OUTPUT
            |---- Conv2dLayer (nn.Module) the convolution layer.
        """
        super(Conv2dLayer, self).__init__()
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, f"Unsupported activation: {activation}"
        if sn:
            # self.conv = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
            #                                    dilation=dilation, bias=bias, padding_mode=padding_mode),
            #                          power_iter=power_iter)
            self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                                         dilation=dilation, bias=bias, padding_mode=padding_mode),
                                               n_power_iterations=power_iter)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                                  bias=bias, padding_mode=padding_mode)

        self.norm = nn.BatchNorm2d(out_channels) if batch_norm else None

    def forward(self, x):
        """
        Forward pass of the GatedConvolution Layer.
        ----------
        INPUT
            |---- x (torch.tensor) input with dimension (Batch x in_Channel x H x W).
        OUTPUT
            |---- out (torch.tensor) the output with dimension (Batch x out_Channel x H' x W').
        """
        # Conv->(BN)->Activation
        out = self.conv(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

class GatedConv2d(nn.Module):
    """
    Define a Gated 2D Convolution as proposed in Yu et al. 2018. (Free-Form Image Inpainting with Gated Convolution).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True,
                 padding_mode='zeros', activation='relu', batch_norm=True):
        """
        Build a Gated 2D convolutional layer as a pytorch nn.Module.
        ----------
        INPUT
            |---- in_channels (int) the number of input channels of the convolutions.
            |---- out_channels (int) the number of output channels of the convolutions.
            |---- kernel_size (int) the kernel size of the convolutions.
            |---- stride (int) the stride of the convolution kernel.
            |---- padding (int) the padding of the input prior to the convolution.
            |---- dilation (int) the dilation of the kernel.
            |---- bias (bool) wether to use a bias term on the convolution kernel.
            |---- padding_mode (str) how to pad the image (see nn.Conv2d doc for details).
            |---- activation (str) the activation function to use. Supported: 'relu' -> ReLU, 'lrelu' -> LeakyReLU,
            |               'prelu' -> PReLU, 'selu' -> SELU, 'tanh' -> Hyperbolic tangent, 'sigmoid' -> sigmoid,
            |               'none' -> No activation used
            |---- batch_norm (bool) whether to use a batch normalization layer between the convolution and the activation.
        OUTPUT
            |---- GatedConv2d (nn.Module) the gated convolution layer.
        """
        super(GatedConv2d, self).__init__()
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, f"Unsupported activation: {activation}"

        self.conv_feat = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                                   bias=bias, padding_mode=padding_mode)
        self.conv_gate = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                                   bias=bias, padding_mode=padding_mode)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.BatchNorm2d(out_channels) if batch_norm else None

    def forward(self, x):
        """
        Forward pass of the GatedConvolution Layer.
        ----------
        INPUT
            |---- x (torch.tensor) input with dimension (Batch x in_Channel x H x W).
        OUTPUT
            |---- out (torch.tensor) the output with dimension (Batch x out_Channel x H' x W').
        """
        # Conv->(BN)->Activation
        feat = self.conv_feat(x)
        if self.norm:
            feat = self.norm(feat)
        if self.activation:
            feat = self.activation(feat)
        # gating
        gate = self.sigmoid(self.conv_gate(x))
        # output
        out = feat * gate
        return out

class UpsampleGatedConv2d(nn.Module):
    """
    Define an upsampling gated block. It combines an upsampling and a gated convolutional layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True,
                 padding_mode='zeros', activation='relu', batch_norm=True, scale_factor=2, size=None,
                 upsampling_mode='nearest', align_corners=None):
        """
        Build a Gated 2D convolutional layer as a pytorch nn.Module.
        ----------
        INPUT
            |---- in_channels (int) the number of input channels of the convolutions.
            |---- out_channels (int) the number of output channels of the convolutions.
            |---- kernel_size (int) the kernel size of the convolutions.
            |---- stride (int) the stride of the convolution kernel.
            |---- padding (int) the padding of the input prior to the convolution.
            |---- dilation (int) the dilation of the kernel.
            |---- bias (bool) wether to use a bias term on the convolution kernel.
            |---- padding_mode (str) how to pad the image (see nn.Conv2d doc for details).
            |---- activation (str) the activation function to use. Supported: 'relu' -> ReLU, 'lrelu' -> LeakyReLU,
            |               'prelu' -> PReLU, 'selu' -> SELU, 'tanh' -> Hyperbolic tangent, 'sigmoid' -> sigmoid,
            |               'none' -> No activation used
            |---- batch_norm (bool) whether to use a batch normalization layer between the convolution and the activation.
            |---- scale_factor (float or tuple of float) the scaling factor to be passed to nn.Upsample. Must be None if
            |               size is specified.
            |---- size (int or tuple of int) the output size of the upsampling. Must be None if scale factor is given.
            |---- upsampling_mode (str) the upsampling interpolation strategy to be passed to nn.Upsample.
            |---- align_corners (bool) whether to align corner pixel of input and output.
        OUTPUT
            |---- UpsampleGatedConv2d (nn.Module) the gated convolution layer.
        """
        super(UpsampleGatedConv2d, self).__init__()
        self.upsample = nn.Upsample(size=size, scale_factor=scale_factor, mode=upsampling_mode, align_corners=align_corners)
        self.gated_conv = GatedConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                      dilation=dilation, bias=bias, padding_mode=padding_mode, activation=activation,
                                      batch_norm=batch_norm)

    def forward(self, x):
        """
        Forward pass of the Upsample + GatedConvolution Layer.
        ----------
        INPUT
            |---- x (torch.tensor) input with dimension (Batch x in_Channel x H x W).
        OUTPUT
            |---- out (torch.tensor) the output with dimension (Batch x out_Channel x H' x W').
        """
        x = self.upsample(x)
        out = self.gated_conv(x)
        return out

class ContextualAttention(nn.Module):
    """
    Yu et al. 2018
    """
    def __init__(self, kernel_size=3, patch_stride=1, compression_rate=1, softmax_scale=10, fuse=False, fuse_kernel=3, device='cuda'):
        """

        """
        super(ContextualAttention, self).__init__()
        self.kernel_size = kernel_size
        self.softmax_scale = softmax_scale
        # reduction param
        self.patch_stride = patch_stride # to compute less weights --> reduce memory impact
        self.compression_rate = compression_rate # to downscale fg and bg --> reduce memory impact
        # fuse param
        self.fuse = fuse
        self.fuse_kernel = fuse_kernel
        # technical param
        self.device = device
        self.eps = 1e-9

    def same_pad(self, im, ksize, stride=1, dilation=1):
        """

        """
        assert im.ndim == 4, f'Expect 4D input, B x C x H x W. Got {im.ndim}'
        ksize = (ksize, ksize) if isinstance(ksize, int) else ksize
        stride = (stride, stride) if isinstance(stride, int) else stride
        dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        # compute padding
        rows, cols = im.shape[2], im.shape[3]
        out_rows = (rows + stride[0] - 1) // stride[0]
        out_cols = (cols + stride[1] - 1) // stride[1]
        effective_k_row = (ksize[0] - 1) * dilation[0] + 1
        effective_k_col = (ksize[1] - 1) * dilation[1] + 1
        padding_rows = max(0, (out_rows - 1) * stride[0] + effective_k_row - rows)
        padding_cols = max(0, (out_cols-1) * stride[1] + effective_k_col - cols)
        # Pad the input
        padding_top = int(padding_rows / 2.)
        padding_left = int(padding_cols / 2.)
        padding_bottom = padding_rows - padding_top
        padding_right = padding_cols - padding_left
        paddings = (padding_left, padding_right, padding_top, padding_bottom)
        return torch.nn.ZeroPad2d(paddings)(im)

    def extract_patch(self, im, ks, stride=1, dilation=1, same_pad=True):
        """

        """
        if same_pad:
            im = self.same_pad(im, ksize=ks, stride=stride, dilation=dilation)
        unfold = nn.Unfold(kernel_size=ks, dilation=dilation, padding=0, stride=stride)
        return unfold(im)

    def forward(self, fg, bg, mask=None):
        """

        """
        # Step 1 extract background patches for reconstruction. Kernel, Stride and
        # Dilation are set to match the downsample version of background/foreground (memory efficecy)
        in_bg_size, in_fg_size = list(bg.shape), list(fg.shape)
        raw_kernel = 2*self.compression_rate
        weight_recon = self.extract_patch(bg, raw_kernel, stride=self.compression_rate*self.patch_stride,
                                          dilation=self.compression_rate, same_pad=True) # [N, C*k*k, L]
        # reshape weights as [N,L,C,k,k] and split along batch dimension
        weight_recon = weight_recon.view(in_bg_size[0], in_bg_size[1], raw_kernel, raw_kernel, -1).permute(0, 4, 1, 2, 3)
        weight_recon_list = torch.split(weight_recon, 1, dim=0)

        # Step 2 Downsample bg and fg to reduce the computation
        fg = F.interpolate(fg, scale_factor=1./self.compression_rate, mode='nearest', recompute_scale_factor=False)
        bg = F.interpolate(bg, scale_factor=1./self.compression_rate, mode='nearest', recompute_scale_factor=False)

        # Step 3 Split fg at level of batch
        fg_list = torch.split(fg, 1, dim=0)

        # Step 4 extract weights for bg<->fg similarity computation
        weight = self.extract_patch(bg, self.kernel_size, stride=self.patch_stride, dilation=1, same_pad=True)
        # reshape weights as [N,L,C,k,k] and split along batch dimension
        weight = weight.view(bg.shape[0], bg.shape[1], self.kernel_size, self.kernel_size, -1).permute(0, 4, 1, 2, 3)
        weight_list = torch.split(weight, 1, dim=0)

        # Step 5 Downscale mask to same dimension as bg and fg & extract patches
        if mask is None:
            mask = torch.ones([bg.shape[0], 1, bg.shape[2], bg.shape[3]], device=self.device)
        else:
            mask = F.interpolate(mask, size=(bg.shape[2], bg.shape[3]), mode='nearest')

        m = self.extract_patch(mask, self.kernel_size, stride=self.patch_stride, dilation=1, same_pad=True)
        # reshape weights as [N,L,C,k,k], take the mean over [C,k,k] and split along batch dimension
        m = m.view(mask.shape[0], mask.shape[1], self.kernel_size, self.kernel_size, -1).permute(0, 4, 1, 2, 3)
        m = torch.mean(m, dim=[2,3,4], keepdim=True).permute(0, 2, 1, 3, 4) # [N, 1, L, 1, 1, 1] <-- importance of patch on forground representation
        m_list = torch.split(m, 1, dim=0)

        # Step 6 Compute similarity between fg and bg for each batch element, then recontruct using reconstruction weights
        sim = []
        for fg_i, weight_i, weight_recon_i, m_i in zip(fg_list, weight_list, weight_recon_list, m_list):
            # Step 6.1 Normalize weights
            weight_i = weight_i.squeeze(0) #remove batch dim
            max_weight_i = torch.sqrt(torch.sum(weight_i ** 2 + self.eps, dim=[1,2,3], keepdim=True))
            weight_i = weight_i / max_weight_i
            # Step 6.2 Compute the similarity of fg with bg patches (a.k.a. weight_i)
            sim_i = F.conv2d(self.same_pad(fg_i, self.kernel_size, stride=1, dilation=1), weight_i, stride=1, padding=0)
            # Step 6.3 Fuse similarity right-left and up-down (--> convolution with ID matrix)
            if self.fuse:
                fuse_weight = torch.eye(self.fuse_kernel, device=self.device).view(1, 1, self.fuse_kernel, self.fuse_kernel)
                n_bg_patch = (int(bg.shape[2]/self.patch_stride), int(bg.shape[3]/self.patch_stride)) # number of bg patch in h and w
                fg_dim = (fg.shape[2], fg.shape[3])
                # first convolution
                sim_i = sim_i.view(1, 1, n_bg_patch[0]*n_bg_patch[1], fg_dim[0]*fg_dim[1])
                sim_i = F.conv2d(self.same_pad(sim_i, self.fuse_kernel, stride=1, dilation=1), fuse_weight, stride=1)
                sim_i = sim_i.contiguous().view(1, n_bg_patch[0], n_bg_patch[1], fg_dim[0], fg_dim[1]).permute(0, 2, 1, 4, 3) # transpose
                # second convolution
                sim_i = sim_i.contiguous().view(1, 1, n_bg_patch[0]*n_bg_patch[1], fg_dim[0]*fg_dim[1])
                sim_i = F.conv2d(self.same_pad(sim_i, self.fuse_kernel, stride=1, dilation=1), fuse_weight, stride=1)
                sim_i = sim_i.contiguous().view(1, n_bg_patch[1], n_bg_patch[0], fg_dim[1], fg_dim[0]).permute(0, 2, 1, 4, 3).contiguous() # transpose back
                # reshape
                sim_i = sim_i.view(1, n_bg_patch[0]*n_bg_patch[1], fg_dim[0], fg_dim[1])

            # Step 6.4 Compute softmax of similarities only on samples of specified as fg in mask
            sim_i = sim_i * m_i.squeeze(0)
            sim_i = F.softmax(sim_i * self.softmax_scale, dim=1)
            sim_i = sim_i * m_i.squeeze(0)

            # Step 6.5 Reconstruct
            weight_recon_i = weight_recon_i.squeeze(0)
            sim_i = F.conv_transpose2d(sim_i, weight_recon_i, stride=self.compression_rate, padding=0) / (raw_kernel ** 2) # average overlapped pixels
            sim_i = sim_i[:, :, :in_fg_size[2], :in_fg_size[3]] # prune extra padding
            sim.append(sim_i)

        # Step 7 reforme the mini-batch
        sim = torch.cat(sim, dim=0).contiguous().view(in_fg_size)  # back to the mini-batch
        return sim

class SelfAttention(nn.Module):
    """
    Self-Attention Module proposed by Zhang et al. 2018 in Self-Attention Generative Adversarial Networks. The module is
    inspired from https://github.com/nihalsid/deepfillv2-pylightning/tree/4b51cbcefaa9d497f5cb2a17d1bd776e4ab2446d where
    the convolution v is removed.
    """
    def __init__(self, in_channels):
        """
        Build a Self-Attention Module.
        ----------
        INPUT
            |---- in_channels (int) the number of channels of the input feature map.
        OUTPUT
            |---- SelfAttention (nn.Module) the Self-Attention modul.
        """
        super(SelfAttention, self).__init__()
        self.conv_f = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1)
        self.conv_g = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1)
        self.conv_h = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Forward pass of the self-attention module.
        INPUT
            |---- x (torch.tensor) the input with dimension [B, in_Channel, H, W].
        OUTPUT
            |---- out (torch.tensor) the output with dimension [B, in_Channel, H, W].
        """
        B, C, H, W = x.size()
        proj_f = self.conv_f(x).view(B, -1, H*W).permute(0, 2, 1)
        proj_g = self.conv_g(x).view(B, -1, H*W)
        attention = self.softmax(torch.bmm(proj_f, proj_g))
        proj_v = self.conv_h(x).view(B, -1, H*W)

        out = torch.bmm(proj_v, attention.permute(0, 2, 1)).view(B, C, H, W)
        out = self.gamma * out + x
        return out

class GatedGenerator(nn.Module):
    """
    Define a gated generator as presented in Yu et al. 2018 (Generative Inpainting Inpainting with Contextual Attention)
    and Yu et al. 2019 (Free Form Image Inpainting with Gated Convolution).
    """
    def __init__(self, in_channels=2, out_channels=1, lat_channels=32, activation='relu', norm=True, padding_mode='reflect',
                 bias=True, upsample_mode='nearest', context_attention=True, return_coarse=True,
                 context_attention_kwargs=dict(kernel_size=3, patch_stride=1, compression_rate=1, softmax_scale=10,
                                               fuse=False, fuse_kernel=3, device='cuda')):
        """
        Build a Gated Generator network.
        ----------
        INPUT
            |---- in_channels (int) the number of input channels (usually 2 for grayscale and 4 for RGB).
            |---- out_channels (int) the number of output channels (usually 1 for grayscale and 3 for RGB).
            |---- lat_channels (int) the number of channels in the first convolution. The number of channels is then
            |               doubled twice at each down sampling and divided by 2 every upsampling.
            |---- activation (str) the activation to use. (possible values : 'relu', 'lrelu', 'prelu', 'selu', 'sigmoid',
            |               'tanh', 'none'). Note that the final activation is a TanH.
            |---- norm (bool) whether to use Batch normalization layer in gated convolution. The first and last convolution
            |               of each Encoder-Decoder does not have BatchNorm layers.
            |---- padding_mode (str) the padding strategy to use. (see pytorch Conv2d doc for list of possible padding modes.)
            |---- bias (bool) whether to include a bias term in the gated convolutions.
            |---- upsample_mode (str) the interpolation strategy in upsampling layers. (see pytorch nn.Upsample doc for
            |               list of possible padding modes.)
            |---- context_attention (bool) whether to include the contextual attention branch.
            |---- return_coarse (bool) whether to return the coarse inpainting results.
        OUTPUT
            |---- GatedGenerator (nn.Module) the generator.
        """
        super(GatedGenerator, self).__init__()
        self.return_coarse = return_coarse
        # Initial coarse Encoder-Decoder
        self.coarse = nn.Sequential(
            GatedConv2d(in_channels,    lat_channels,   5, stride=1, dilation=1,  padding=2,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=False),
            GatedConv2d(lat_channels,   2*lat_channels, 3, stride=2, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(2*lat_channels, 2*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(2*lat_channels, 4*lat_channels, 3, stride=2, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=2,  padding=2,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=4,  padding=4,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=8,  padding=8,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=16, padding=16, padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            UpsampleGatedConv2d(4*lat_channels, 2*lat_channels, 3, stride=1, dilation=1, padding=1, padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm,
                                scale_factor=2, upsampling_mode=upsample_mode),
            GatedConv2d(2*lat_channels, 2*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            UpsampleGatedConv2d(2*lat_channels, lat_channels, 3, stride=1, dilation=1, padding=1, padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm,
                                scale_factor=2, upsampling_mode=upsample_mode),
            GatedConv2d(lat_channels, lat_channels//2, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(lat_channels//2, out_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation='sigmoid',     bias=bias, batch_norm=False)
        )
        # Encoder of refinement
        self.refine_enc = nn.Sequential(
            GatedConv2d(in_channels,    lat_channels,   5, stride=1, dilation=1,  padding=2,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=False),
            GatedConv2d(lat_channels,   2*lat_channels, 3, stride=2, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(2*lat_channels, 2*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(2*lat_channels, 4*lat_channels, 3, stride=2, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=2,  padding=2,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=4,  padding=4,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=8,  padding=8,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=16, padding=16, padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm)
        )
        # Optional Contextual Attention branch of refinement encoder
        self.refine_attention_enc = nn.ModuleDict({
            'cnn1': nn.Sequential(
                        GatedConv2d(in_channels,    lat_channels,   5, stride=1, dilation=1,  padding=2,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=False),
                        GatedConv2d(lat_channels,   2*lat_channels, 3, stride=2, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
                        GatedConv2d(2*lat_channels, 2*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
                        GatedConv2d(2*lat_channels, 4*lat_channels, 3, stride=2, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
                        GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
                        GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm)),
            'ctx': ContextualAttention(**context_attention_kwargs),
            'cnn2': nn.Sequential(
                        GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
                        GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm))
        }) if context_attention else None
        # Upsample refinement
        in_channels_up = 8*lat_channels if context_attention else 4*lat_channels # define input channels of Up module
        self.refine_dec = nn.Sequential(
            GatedConv2d(in_channels_up, 4*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            UpsampleGatedConv2d(4*lat_channels, 2*lat_channels, 3, stride=1, dilation=1, padding=1, padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm,
                                scale_factor=2, upsampling_mode=upsample_mode),
            GatedConv2d(2*lat_channels, 2*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            UpsampleGatedConv2d(2*lat_channels, lat_channels, 3, stride=1, dilation=1, padding=1, padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm,
                                scale_factor=2, upsampling_mode=upsample_mode),
            GatedConv2d(lat_channels, lat_channels//2, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(lat_channels//2, out_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation='sigmoid',     bias=bias, batch_norm=False)
        )

    def forward(self, img, mask):
        """
        mask = 0 where valid pixel and 1 on region to inpaint
        Inpaint the passed image on area specified my the mask.
        ----------
        INPUT
            |---- img (torch.tensor) the image to inpaint with dimension (Batch x In-Channels-1 x H x W)
            |---- mask (torch.tensor) the mask of region to inpaint. Region to inpaint must be sepcified by 1 and region
            |               kept untouched must be 0. The mask should have dimension (Batch x 1 x H x W).
        OUTPUT
            |---- fine_inpaint (torch.tensor) the fine inpainted image with dimension (Batch x In-Channels x H x W).
            |---- (coarse_inpaint) (torch.tensor) the intermediate coarse inpaint with dimension (Batch x In-Channels x H x W).
        """
        masked_img = img * (1 - mask) # mask = 1 on region to remove
        # stage 1 - Coarse
        input = torch.cat([masked_img, mask], dim=1)
        coarse_inpaint = self.coarse(input)
        coarse_inpaint_corr = coarse_inpaint * mask + masked_img # keep inpaint only on region to inpaint
        # stage 2 - Refinement
        input_2 = torch.cat([coarse_inpaint_corr, mask], dim=1)
        x = self.refine_enc(input_2)

        if self.refine_attention_enc is not None:
            #x_context = self.refine_attention_enc(input_2)
            x_context = self.refine_attention_enc['cnn1'](input_2)
            x_context = self.refine_attention_enc['ctx'](x_context, x_context, mask=mask)
            x_context = self.refine_attention_enc['cnn2'](x_context)
            x = torch.cat([x, x_context], dim=1)

        fine_inpaint = self.refine_dec(x)
        #fine_inpaint = fine_inpaint * mask + masked_img # keep inpaint only on region to inpaint
        # return inpainted image
        if self.return_coarse:
            return fine_inpaint, coarse_inpaint
        else:
            return fine_inpaint

class PatchDiscriminator(nn.Module):
    """
    Define a Patch Discriminator proposed in Yu et al. (Free-Form Image Inpainting with Gated Convolution) for image
    inpainting. with a possible Self-attention layer from Zhang et al 2018.
    """
    def __init__(self, in_channels=2, out_channels=[64, 128, 256, 256, 256, 256], kernel_size=5, stride=2, bias=True,
                 activation='relu', norm=True, padding_mode='zeros', sn=True, self_attention=True):
        """
        Build a PatchDiscriminator.
        ----------
        INPUT
            |---- in_channels (int) the number of channels of input. (image channel + mask channel (=1))
            |---- out_channels (list of int) the number of output channels of each convolutinal layer. The lenght of this
            |               list defines the depth of the network.
            |---- kernel_size (int or list of int) the kernel size to use in convolutions. If a int is provided, the same
            |               kernel size will be used for all layers. If a list is passed, it must be of the same length
            |               as out_channels.
            |---- stride (int or list of int) the convolution stride. If a int is passed, the same stride is applied in
            |               each layer excpet the first one with sride=1. If a list, it must be of same size as out_channels.
            |---- bias (bool or list of bool) whether to include a bias term on convolution. If a bool is passed, bias is
            |               included in each layer. If a list, it must be of same size as out_channels.
            |---- activation (str or list of string) activation function to use. If a str is passed, all layer expect the
            |               last one are activated with the same passed activation. If a list, it must be of same size
            |               as out_channels. (possible values : 'relu', 'lrelu', 'prelu', 'selu', 'sigmoid', 'tanh', 'none').
            |---- norm (bool or list of bool) whether to use a Batchnorm layer in convolution. If a bool is passed all
            |               layers have the same norm. If a list, it must be of same size as out_channels.
            |---- padding_mode (str or list of str) how to pad the features map. If a str is passed all layers have the
            |               same padding mode. If a list, it must be of same size as out_channels. (see pytorch Conv2d
            |               doc for list of possible padding modes.)
            |---- sn (bool or list of bool) whether to apply Spectral Normalization to the convolution layers. If a bool
            |               is passed all layers have/have not the Spectral Normalization. If a list, it must be of same
            |               size as out_channels.
            |---- self_attention (bool) whether to add a self-attention layer before the last convolution
        OUTPUT
            |---- PatchDiscriminator (nn.Module) the patch discriminator.
        """
        super(PatchDiscriminator, self).__init__()
        n_layer = len(out_channels)
        in_channels = [in_channels] + out_channels[:-1]
        if isinstance(activation, list):
            assert len(activation) == n_layer, f"Activation provided as list but does not match the number of layers. Given {len(activation)} ; required {n_layer}"
        else:
            activation = [activation] * (n_layer-1) + ['none']
        if isinstance(kernel_size, list):
            assert len(kernel_size) == n_layer, f"Kernel sizes provided as list but does not match the number of layers. Given {len(kernel_size)} ; required {n_layer}"
        else:
            kernel_size = [kernel_size] * n_layer
        if isinstance(stride, list):
            assert len(stride) == n_layer, f"Stride provided as list but does not match the number of layers. Given {len(stride)} ; required {n_layer}"
        else:
            stride = [1] + [stride] * (n_layer - 1)
        if isinstance(bias, list):
            assert len(bias) == n_layer, f"Bias provided as list but does not match the number of layers. Given {len(bias)} ; required {n_layer}"
        else:
            bias = [bias] * n_layer
        if isinstance(padding_mode, list):
            assert len(padding_mode) == n_layer, f"Padding Mode provided as list but does not match the number of layers. Given {len(padding_mode)} ; required {n_layer}"
        else:
            padding_mode = [padding_mode] * n_layer
        if isinstance(sn, list):
            assert len(sn) == n_layer, f"Spectral Normalization provided as list but does not match the number of layers. Given {len(sn)} ; required {n_layer}"
        else:
            sn = [sn] * n_layer
        if isinstance(norm, list):
            assert len(norm) == n_layer, f"BatchNormalization provided as list but does not match the number of layers. Given {len(norm)} ; required {n_layer}"
        else:
            norm = [norm] * n_layer
        padding = [(ks - 1) // 2 for ks in kernel_size]
        # build conv_layers
        self.layer_list = nn.ModuleList()
        for i in range(n_layer):
            self.layer_list.append(Conv2dLayer(in_channels[i], out_channels[i], kernel_size[i], stride=stride[i],
                                               padding=padding[i], bias=bias[i], padding_mode=padding_mode[i],
                                               activation=activation[i], batch_norm=norm[i], sn=sn[i]))
            if self_attention and i == (n_layer - 2):
                self.layer_list.append(SelfAttention(in_channels=out_channels[i]))
                self.layer_list.append(nn.ReLU())

    def forward(self, img, mask):
        """
        Forward pass of the patch discriminator.
        ----------
        INPUT
            |---- img (torch.tensor) the image inpainted or not with dimension (Batch x In-Channels-1 x H x W)
            |---- mask (torch.tensor) the mask of region inpainted. Region inpainted must be sepcified by 1 and region
            |               kept untouched must be 0. The mask should have dimension (Batch x 1 x H x W).
        OUTPUT
            |---- x (torch.tensor) the output feature map with dimension (Batch x Out-Channels x H' x W').
        """
        # concat img and mask
        x = torch.cat([img, mask], dim=1)
        # CNN
        for layer in self.layer_list:
            x = layer(x)
        return x

class SAGatedGenerator(nn.Module):
    """
    Define a self-attention gated generator as presented in Yu et al. 2018 (Generative Inpainting Inpainting with Contextual Attention)
    and Yu et al. 2019 (Free Form Image Inpainting with Gated Convolution) but with the self-attention module of Zhang et
    al. 2018 instead of the contextual attention module.
    """
    def __init__(self, in_channels=2, out_channels=1, lat_channels=32, activation='relu', norm=True, padding_mode='reflect',
                 bias=True, upsample_mode='nearest', self_attention=True, return_coarse=True):
        """
        Build a Self-Attention Gated Generator network.
        ----------
        INPUT
            |---- in_channels (int) the number of input channels (usually 2 for grayscale and 4 for RGB).
            |---- out_channels (int) the number of output channels (usually 1 for grayscale and 3 for RGB).
            |---- lat_channels (int) the number of channels in the first convolution. The number of channels is then
            |               doubled twice at each down sampling and divided by 2 every upsampling.
            |---- activation (str) the activation to use. (possible values : 'relu', 'lrelu', 'prelu', 'selu', 'sigmoid',
            |               'tanh', 'none'). Note that the final activation is a TanH.
            |---- norm (bool) whether to use Batch normalization layer in gated convolution. The first and last convolution
            |               of each Encoder-Decoder does not have BatchNorm layers.
            |---- padding_mode (str) the padding strategy to use. (see pytorch Conv2d doc for list of possible padding modes.)
            |---- bias (bool) whether to include a bias term in the gated convolutions.
            |---- upsample_mode (str) the interpolation strategy in upsampling layers. (see pytorch nn.Upsample doc for
            |               list of possible padding modes.)
            |---- self_attention (bool) whether to include the self attention module before refinement upsampling.
            |---- return_coarse (bool) whether to return the coarse inpainting results.
        OUTPUT
            |---- GatedGenerator (nn.Module) the generator.
        """
        super(SAGatedGenerator, self).__init__()
        self.return_coarse = return_coarse
        # Initial coarse Encoder-Decoder
        self.coarse = nn.Sequential(
            GatedConv2d(in_channels,    lat_channels,   5, stride=1, dilation=1,  padding=2,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=False),
            GatedConv2d(lat_channels,   2*lat_channels, 3, stride=2, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(2*lat_channels, 2*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(2*lat_channels, 4*lat_channels, 3, stride=2, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=2,  padding=2,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=4,  padding=4,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=8,  padding=8,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=16, padding=16, padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            UpsampleGatedConv2d(4*lat_channels, 2*lat_channels, 3, stride=1, dilation=1, padding=1, padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm,
                                scale_factor=2, upsampling_mode=upsample_mode),
            GatedConv2d(2*lat_channels, 2*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            UpsampleGatedConv2d(2*lat_channels, lat_channels, 3, stride=1, dilation=1, padding=1, padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm,
                                scale_factor=2, upsampling_mode=upsample_mode),
            GatedConv2d(lat_channels, lat_channels//2, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(lat_channels//2, out_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation='sigmoid',     bias=bias, batch_norm=False)
        )
        # Encoder of refinement
        self.refine_enc = nn.Sequential(
            GatedConv2d(in_channels,    lat_channels,   5, stride=1, dilation=1,  padding=2,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=False),
            GatedConv2d(lat_channels,   2*lat_channels, 3, stride=2, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(2*lat_channels, 2*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(2*lat_channels, 4*lat_channels, 3, stride=2, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=2,  padding=2,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=4,  padding=4,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=8,  padding=8,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=16, padding=16, padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm)
        )

        self.refine_attention = nn.Sequential(SelfAttention(4*lat_channels), nn.ReLU()) if self_attention else None

        self.refine_dec = nn.Sequential(
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(4*lat_channels, 4*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            UpsampleGatedConv2d(4*lat_channels, 2*lat_channels, 3, stride=1, dilation=1, padding=1, padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm,
                                scale_factor=2, upsampling_mode=upsample_mode),
            GatedConv2d(2*lat_channels, 2*lat_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            UpsampleGatedConv2d(2*lat_channels, lat_channels, 3, stride=1, dilation=1, padding=1, padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm,
                                scale_factor=2, upsampling_mode=upsample_mode),
            GatedConv2d(lat_channels, lat_channels//2, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation=activation, bias=bias, batch_norm=norm),
            GatedConv2d(lat_channels//2, out_channels, 3, stride=1, dilation=1,  padding=1,  padding_mode=padding_mode, activation='sigmoid',     bias=bias, batch_norm=False)
        )

    def forward(self, img, mask):
        """
        Inpaint the passed image on area specified my the mask.
        ----------
        INPUT
            |---- img (torch.tensor) the image to inpaint with dimension (Batch x In-Channels-1 x H x W)
            |---- mask (torch.tensor) the mask of region to inpaint. Region to inpaint must be sepcified by 1 and region
            |               kept untouched must be 0. The mask should have dimension (Batch x 1 x H x W).
        OUTPUT
            |---- fine_inpaint (torch.tensor) the fine inpainted image with dimension (Batch x In-Channels x H x W).
            |---- (coarse_inpaint) (torch.tensor) the intermediate coarse inpaint with dimension (Batch x In-Channels x H x W).
        """
        masked_img = img * (1 - mask) # mask = 1 on region to remove

        # stage 1 - Coarse
        input = torch.cat([masked_img, mask], dim=1)
        coarse_inpaint = self.coarse(input)
        coarse_inpaint_corr = coarse_inpaint * mask + masked_img # keep inpaint only on region to inpaint

        # stage 2 - Refinement
        input_2 = torch.cat([coarse_inpaint_corr, mask], dim=1)
        x = self.refine_enc(input_2)
        # self-attention
        if self.refine_attention is not None:
            x = self.refine_attention(x)
        # decoding
        fine_inpaint = self.refine_dec(x)
        # return inpainted image
        if self.return_coarse:
            return fine_inpaint, coarse_inpaint
        else:
            return fine_inpaint
