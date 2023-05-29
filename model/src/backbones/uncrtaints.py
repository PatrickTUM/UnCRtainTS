"""
UnCRtainTS Implementation
Author: Patrick Ebel (github/patrickTUM)
License: MIT
"""

import torch
import torch.nn as nn

from src.backbones.utae import ConvLayer, ConvBlock, TemporallySharedBlock
from src.backbones.ltae import LTAE2d, LTAE2dtiny

S2_BANDS = 13


def get_norm_layer(out_channels, num_feats, n_groups=4, layer_type='batch'):
    if layer_type == 'batch':
        return nn.BatchNorm2d(out_channels)
    elif layer_type == 'instance':
        return nn.InstanceNorm2d(out_channels)
    elif layer_type == 'group':
        return nn.GroupNorm(num_channels=num_feats, num_groups=n_groups)

class ResidualConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        nkernels,
        pad_value=None,
        norm="batch",
        n_groups=4,
        #last_relu=True,
        k=3, s=1, p=1,
        padding_mode="reflect",
    ):
        super(ResidualConvBlock, self).__init__(pad_value=pad_value)

        self.conv1 = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=True,
            k=k, s=s, p=p,
            n_groups=n_groups,
            padding_mode=padding_mode,
        )
        self.conv2 = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=True,
            k=k, s=s, p=p,
            n_groups=n_groups,
            padding_mode=padding_mode,
        )
        self.conv3 = ConvLayer(
            nkernels=nkernels,
            #norm='none',
            #last_relu=False,
            norm=norm,
            last_relu=True,
            k=k, s=s, p=p,
            n_groups=n_groups,
            padding_mode=padding_mode,
        )

    def forward(self, input):

        out1 = self.conv1(input)        # followed by built-in ReLU & norm
        out2 = self.conv2(out1)         # followed by built-in ReLU & norm
        out3 = input + self.conv3(out2) # omit norm & ReLU
        return out3


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm, n_groups=4):
        super().__init__()
        self.norm = get_norm_layer(dim, dim, n_groups, norm)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

    
class MBConv(TemporallySharedBlock):
    def __init__(self, inp, oup, downsample=False, expansion=4, norm='batch', n_groups=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, stride=1, padding=0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride,
                          padding=1, padding_mode='reflect', groups=hidden_dim, bias=False),
                get_norm_layer(hidden_dim, hidden_dim, n_groups, norm),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, stride=1, padding=0, bias=False),
                get_norm_layer(oup, oup, n_groups, norm),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride=stride, padding=0, bias=False),
                get_norm_layer(hidden_dim, hidden_dim, n_groups, norm),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1, padding_mode='reflect',
                          groups=hidden_dim, bias=False),
                get_norm_layer(hidden_dim, hidden_dim, n_groups, norm),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, stride=1, padding=0, bias=False),
                get_norm_layer(oup, oup, n_groups, norm), 
            )
        
        self.conv = PreNorm(inp, self.conv, norm, n_groups=4)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)


class Compact_Temporal_Aggregator(nn.Module):
    def __init__(self, mode="mean"):
        super(Compact_Temporal_Aggregator, self).__init__()
        self.mode = mode
        # moved dropout from ScaledDotProductAttention to here, applied after upsampling 
        self.attn_dropout = nn.Dropout(0.1) # no dropout via: nn.Dropout(0.0)

    def forward(self, x, pad_mask=None, attn_mask=None):
        if pad_mask is not None and pad_mask.any():
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)

                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                    # this got moved out of ScaledDotProductAttention, apply after upsampling
                    attn = self.attn_dropout(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)

                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                attn = attn * (~pad_mask).float()[None, :, :, None, None]

                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                # this got moved out of ScaledDotProductAttention, apply after upsampling
                attn = self.attn_dropout(attn)
                attn = attn * (~pad_mask).float()[:, :, None, None]
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                out = x * (~pad_mask).float()[:, :, None, None, None]
                out = out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
                return out
        else:
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)
                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                    # this got moved out of ScaledDotProductAttention, apply after upsampling
                    attn = self.attn_dropout(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                # this got moved out of ScaledDotProductAttention, apply after upsampling
                attn = self.attn_dropout(attn)
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                return x.mean(dim=1)

def get_nonlinearity(mode, eps):
    if mode=='relu':        fct = nn.ReLU() + eps 
    elif mode=='softplus':  fct = lambda vars:nn.Softplus(beta=1, threshold=20)(vars) + eps
    elif mode=='elu':       fct = lambda vars: nn.ELU()(vars) + 1 + eps  
    else:                   fct = nn.Identity()
    return fct

class UNCRTAINTS(nn.Module):
    def __init__(
        self,
        input_dim,
        encoder_widths=[128],
        decoder_widths=[128,128,128,128,128],
        out_conv=[S2_BANDS],
        out_nonlin_mean=False,
        out_nonlin_var='relu',
        agg_mode="att_group",
        encoder_norm="group",
        decoder_norm="batch",
        n_head=16,
        d_model=256,
        d_k=4,
        pad_value=0,
        padding_mode="reflect",
        positional_encoding=True,
        covmode='diag',
        scale_by=1,
        separate_out=False,
        use_v=False,
        block_type='mbconv',
        is_mono=False
    ):
        """
        UnCRtainTS architecture for spatio-temporal encoding of satellite image time series.
        Args:
            input_dim (int): Number of channels in the input images.
            encoder_widths (List[int]): List giving the number of channels of the successive encoder_widths of the convolutional encoder.
            This argument also defines the number of encoder_widths (i.e. the number of downsampling steps +1)
            in the architecture.
            The number of channels are given from top to bottom, i.e. from the highest to the lowest resolution.
            decoder_widths (List[int], optional): Same as encoder_widths but for the decoder. The order in which the number of
            channels should be given is also from top to bottom. If this argument is not specified the decoder
            will have the same configuration as the encoder.
            out_conv (List[int]): Number of channels of the successive convolutions for the
            agg_mode (str): Aggregation mode for the skip connections. Can either be:
                - att_group (default) : Attention weighted temporal average, using the same
                channel grouping strategy as in the LTAE. The attention masks are bilinearly
                resampled to the resolution of the skipped feature maps.
                - att_mean : Attention weighted temporal average,
                 using the average attention scores across heads for each date.
                - mean : Temporal average excluding padded dates.
            encoder_norm (str): Type of normalisation layer to use in the encoding branch. Can either be:
                - group : GroupNorm (default)
                - batch : BatchNorm
                - instance : InstanceNorm
                - none: apply no normalization
            decoder_norm (str): similar to encoder_norm
            n_head (int): Number of heads in LTAE.
            d_model (int): Parameter of LTAE
            d_k (int): Key-Query space dimension
            pad_value (float): Value used by the dataloader for temporal padding.
            padding_mode (str): Spatial padding strategy for convolutional layers (passed to nn.Conv2d).
            positional_encoding (bool): If False, no positional encoding is used (default True).
        """
        super(UNCRTAINTS, self).__init__()
        self.n_stages       = len(encoder_widths)
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.out_widths     = out_conv
        self.is_mono        = is_mono
        self.use_v          = use_v
        self.block_type     = block_type

        self.enc_dim        = decoder_widths[0] if decoder_widths is not None else encoder_widths[0]
        self.stack_dim      = sum(decoder_widths) if decoder_widths is not None else sum(encoder_widths)
        self.pad_value      = pad_value
        self.padding_mode   = padding_mode

        self.scale_by       = scale_by
        self.separate_out   = separate_out # define two separate layer streams for mean and variance predictions

        if decoder_widths is not None:
            assert encoder_widths[-1] == decoder_widths[-1]
        else: decoder_widths = encoder_widths


        # ENCODER
        self.in_conv = ConvBlock(
            nkernels=[input_dim] + [encoder_widths[0]],
            k=1, s=1, p=0,
            norm=encoder_norm,
        )

        if self.block_type=='mbconv':
            self.in_block = nn.ModuleList([MBConv(layer, layer, downsample=False, expansion=2, norm=encoder_norm) for layer in encoder_widths])
        elif self.block_type=='residual':
            self.in_block = nn.ModuleList([ResidualConvBlock(nkernels=[layer]+[layer], k=3, s=1, p=1, norm=encoder_norm, n_groups=4) for layer in encoder_widths])
        else: raise NotImplementedError

        if not self.is_mono:
            # LTAE
            if self.use_v:
                # same as standard LTAE, except we don't apply dropout on the low-resolution attention masks
                self.temporal_encoder = LTAE2d(
                    in_channels=encoder_widths[0], 
                    d_model=d_model,
                    n_head=n_head,
                    mlp=[d_model, encoder_widths[0]], # MLP to map v, only used if self.use_v=True
                    return_att=True,
                    d_k=d_k,
                    positional_encoding=positional_encoding,
                    use_dropout=False
                )
                # linearly combine mask-weighted
                v_dim = encoder_widths[0]
                self.include_v = nn.Conv2d(encoder_widths[0]+v_dim, encoder_widths[0], 1)
            else:
                self.temporal_encoder = LTAE2dtiny(
                    in_channels=encoder_widths[0],
                    d_model=d_model,
                    n_head=n_head,
                    d_k=d_k,
                    positional_encoding=positional_encoding,
                )
            
            self.temporal_aggregator = Compact_Temporal_Aggregator(mode=agg_mode)

        if self.block_type=='mbconv':
            self.out_block = nn.ModuleList([MBConv(layer, layer, downsample=False, expansion=2, norm=decoder_norm) for layer in decoder_widths])
        elif self.block_type=='residual':
            self.out_block = nn.ModuleList([ResidualConvBlock(nkernels=[layer]+[layer], k=3, s=1, p=1, norm=decoder_norm, n_groups=4) for layer in decoder_widths])
        else: raise NotImplementedError


        self.covmode = covmode
        if covmode=='uni':
            # batching across channel dimension
            covar_dim = S2_BANDS
        elif covmode=='iso':
            covar_dim = 1
        elif covmode=='diag':
            covar_dim = S2_BANDS
        else: covar_dim = 0 

        self.mean_idx = S2_BANDS
        self.vars_idx = self.mean_idx + covar_dim

        # note: not including normalization layer and ReLU nonlinearity into the final ConvBlock
        #       if inserting >1 layers into out_conv then consider treating normalizations separately
        self.out_dims = out_conv[-1]

        eps = 1e-9 if self.scale_by==1.0 else 1e-3

        if self.separate_out: # define two separate layer streams for mean and variance predictions
            self.out_conv_mean_1 = ConvBlock(nkernels=[decoder_widths[0]] + [S2_BANDS], k=1, s=1, p=0, norm='none', last_relu=False)
            if self.out_dims - self.mean_idx > 0:
                self.out_conv_var_1 = ConvBlock(nkernels=[decoder_widths[0]] + [self.out_dims - S2_BANDS], k=1, s=1, p=0, norm='none', last_relu=False)
        else: 
            self.out_conv = ConvBlock(nkernels=[decoder_widths[0]] + out_conv, k=1, s=1, p=0, norm='none', last_relu=False)

        # set output nonlinearities
        if out_nonlin_mean: self.out_mean  = lambda vars: self.scale_by * nn.Sigmoid()(vars)    # this is for predicting mean values in [0, 1]
        else: self.out_mean  = nn.Identity()                                                    # just keep the mean estimates, without applying a nonlinearity

        if self.covmode in ['uni', 'iso', 'diag']:
            self.diag_var   = get_nonlinearity(out_nonlin_var, eps)


    def forward(self, input, batch_positions=None):
        pad_mask = (
            (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # BxT pad mask
        # SPATIAL ENCODER
        # collect feature maps in list 'feature_maps'
        out = self.in_conv.smart_forward(input)

        for layer in self.in_block:
            out = layer.smart_forward(out)

        if not self.is_mono:
            att_down = 32
            down = nn.AdaptiveMaxPool2d((att_down, att_down))(out.view(out.shape[0] * out.shape[1], *out.shape[2:])).view(out.shape[0], out.shape[1], out.shape[2], att_down, att_down)

            # TEMPORAL ENCODER
            if self.use_v:
                v, att = self.temporal_encoder(down, batch_positions=batch_positions, pad_mask=pad_mask)
            else:
                att = self.temporal_encoder(down, batch_positions=batch_positions, pad_mask=pad_mask)

            out = self.temporal_aggregator(out, pad_mask=pad_mask, attn_mask=att)

            if self.use_v:
                # upsample values to input resolution, then linearly combine with attention masks
                up_v = nn.Upsample(size=(out.shape[-2:]), mode="bilinear", align_corners=False)(v)
                out  = self.include_v(torch.cat((out, up_v), dim=1)) 
        else: out = out.squeeze(dim=1)

        # SPATIAL DECODER
        for layer in self.out_block:
            out = layer.smart_forward(out)

        if self.separate_out:
            out_mean_1 = self.out_conv_mean_1(out)

            if self.out_dims - self.mean_idx > 0:
                out_var_1 = self.out_conv_var_1(out)
                out   = torch.cat((out_mean_1, out_var_1), dim=1)
            else: out = out_mean_1 #out = out_mean_2
        else:
            out = self.out_conv(out) # predict mean and var in single layer
        

        # append a singelton temporal dimension such that outputs are [B x T=1 x C x H x W]
        out = out.unsqueeze(dim=1)

        # apply output nonlinearities

        # get mean predictions
        out_loc   = self.out_mean(out[:,:,:self.mean_idx,...])                      # mean predictions in [0,1]
        if not self.covmode: return out_loc

        out_cov = self.diag_var(out[:,:,self.mean_idx:self.vars_idx,...])           # var predictions > 0
        out     = torch.cat((out_loc, out_cov), dim=2)                              # stack mean and var predictions plus cloud masks
        
        return out