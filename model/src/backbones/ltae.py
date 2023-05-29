import copy

import numpy as np
import torch
import torch.nn as nn

from src.backbones.positional_encoding import PositionalEncoder


class LTAE2d(nn.Module):
    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=4,
        mlp=[256, 128],
        dropout=0.2,
        d_model=256,
        T=1000,
        return_att=False,
        positional_encoding=True,
        use_dropout=True
    ):
        """
        Lightweight Temporal Attention Encoder (L-TAE) for image time series.
        Attention-based sequence encoding that maps a sequence of images to a single feature map.
        A shared L-TAE is applied to all pixel positions of the image sequence.
        Args:
            in_channels (int): Number of channels of the input embeddings.
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            mlp (List[int]): Widths of the layers of the MLP that processes the concatenated outputs of the attention heads.
            dropout (float): dropout on the MLP-processed values
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model.
            T (int): Period to use for the positional encoding.
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
            positional_encoding (bool): If False, no positional encoding is used (default True).
            use_dropout (bool): dropout on the attention masks.
        """
        super(LTAE2d, self).__init__()
        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None
        assert self.mlp[0] == self.d_model

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None

        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model, use_dropout=use_dropout
        )
        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.in_channels,
        )
        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=mlp[-1],
        )

        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    nn.BatchNorm1d(self.mlp[i + 1]),
                    nn.ReLU(),
                ]
            )

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None, pad_mask=None, return_comp=False):
        sz_b, seq_len, d, h, w = x.shape
        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            pad_mask = (
                pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            )

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)

        # re-shaped attn to   [h x B*H*W x T], e.g. torch.Size([16, 2048, 4])
        #   in utae.py this is torch.Size([h, B, T, 32, 32])
        # re-shaped output to [h x B*H*W x d_in/h], e.g. torch.Size([16, 2048, 16])
        #   in utae.py this is torch.Size([B, 128, 32, 32])
        out, attn = self.attention_heads(out, pad_mask=pad_mask)

        out = (
            out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1)
        )  # Concatenate heads, out is now [B*H*W x d_in/h * h], e.g. [2048 x 256]

        # out is of shape [head x b x t x h x w]
        out = self.dropout(self.mlp(out))
        # after MLP, out is of shape [B*H*W x outputLayerOfMLP], e.g. [2048 x 128]
        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)
        
        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(
            0, 1, 4, 2, 3
        )

        # out  is of shape [B x outputLayerOfMLP x h x w], e.g. [2, 128, 32, 32]
        # attn is of shape [h x B x T x H x W], e.g. [16, 2, 4, 32, 32]
        if self.return_att:
            return out, attn
        else:
            return out



class LTAE2dtiny(nn.Module):
    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=4,
        d_model=256,
        T=1000,
        positional_encoding=True,
    ):
        """
        Lightweight Temporal Attention Encoder (L-TAE) for image time series.
        Attention-based sequence encoding that maps a sequence of images to a single feature map.
        A shared L-TAE is applied to all pixel positions of the image sequence.
        This is the tiny version, which stops further processing attention-weighted values v
        (no longer using an MLP) and only returns the attention matrix attn itself
        Args:
            in_channels (int): Number of channels of the input embeddings.
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model.
            T (int): Period to use for the positional encoding.
            positional_encoding (bool): If False, no positional encoding is used (default True).
        """
        super(LTAE2dtiny, self).__init__()
        self.in_channels = in_channels
        self.n_head = n_head

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None

        self.attention_heads = MultiHeadAttentionSmall(
            n_head=n_head, d_k=d_k, d_in=self.d_model
        )
        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.in_channels,
        )


    def forward(self, x, batch_positions=None, pad_mask=None):
        sz_b, seq_len, d, h, w = x.shape
        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            pad_mask = (
                pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            )

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            bp  = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)

        # re-shaped attn to   [h x B*H*W x T], e.g. torch.Size([16, 2048, 4])
        #   in utae.py this is torch.Size([h, B, T, 32, 32])
        # re-shaped output to [h x B*H*W x d_in/h], e.g. torch.Size([16, 2048, 16])
        #   in utae.py this is torch.Size([B, 128, 32, 32])
        attn = self.attention_heads(out, pad_mask=pad_mask)


        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(
            0, 1, 4, 2, 3
        )

        # out  is of shape [B x outputLayerOfMLP x h x w], e.g. [2, 128, 32, 32]
        # attn is of shape [h x B x T x H x W], e.g. [16, 2, 4, 32, 32]
        return attn


# this class still uses ScaledDotProductAttention (including dropout)
# and always computes and returns att*v
class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head, d_k, d_in, use_dropout=True):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in # e.g. self.d_model in LTAE2d

        # define H x k queries, they are input-independent in LTAE
        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        attn_dropout=0.1 if use_dropout else 0.0
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=attn_dropout)

    def forward(self, v, pad_mask=None, return_comp=False):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        # values v are of shapes [B*H*W, T, self.d_in=self.d_model], e.g. [2*32*32=2048 x 4 x 256] (see: sz_b * h * w, seq_len, d)
        # where self.d_in=self.d_model is the output dimension of the FC-projected features  
        sz_b, seq_len, _ = v.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(-1, d_k)  # (n*b) x d_k

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (n_head, 1)
            )  # replicate pad_mask for each head (nxb) x lk

        # attn   is of shape [B*H*W*h, 1, T], e.g. [2*32*32*16=32768 x 1 x 4], e.g. Size([32768, 1, 4])
        # v      is of shape [B*H*W*h, T, self.d_in/h], e.g. [2*32*32*16=32768 x 4 x 256/16=16], e.g. Size([32768, 4, 16])
        # output is of shape [B*H*W*h, 1, h], e.g. [2*32*32*16=32768 x 1 x 16], e.g. Size([32768, 1, 16])
        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(n_head * sz_b, seq_len, -1)
        if return_comp:
            output, attn, comp = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        else:
            output, attn = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )

        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        output = output.view(n_head, sz_b, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        # re-shaped attn to   [h x B*H*W x T], e.g. torch.Size([16, 2048, 4])
        #   in utae.py this is torch.Size([h, B, T, 32, 32])
        # re-shaped output to [h x B*H*W x d_in/h], e.g. torch.Size([16, 2048, 16])
        #   in utae.py this is torch.Size([B, 128, 32, 32])
        if return_comp:
            return output, attn, comp
        else:
            return output, attn


# this class uses ScaledDotProductAttentionSmall (excluding dropout)
# and only optionally computes and returns att*v
class MultiHeadAttentionSmall(nn.Module):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head    # e.g. 16
        self.d_k    = d_k       # e.g. 4, number of keys per head
        self.d_in   = d_in      # e.g. 256, self.d_model in LTAE2d

        # define H x k queries, they are input-independent in LTAE
        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        """
        # consider using deeper mappings with nonlinearities,
        # but this is somewhat against the original Transformer spirit
        self.fc1_k = nn.Linear(d_in, d_in)
        self.bn2_k = nn.BatchNorm1d(d_in)
        self.fc2_k = nn.Linear(d_in, n_head * d_k)
        self.bn2_k = nn.BatchNorm1d(n_head * d_k)
        """

        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))
        #nn.init.normal_(self.fc2_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))
        self.attention = ScaledDotProductAttentionSmall(temperature=np.power(d_k, 0.5))

    def forward(self, v, pad_mask=None, return_comp=False, weight_v=False):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        # values v are of shapes [B*H*W, T, self.d_in=self.d_model], e.g. [2*32*32=2048 x 4 x 256] (see: sz_b * h * w, seq_len, d)
        # where self.d_in=self.d_model is the output dimension of the FC-projected features  
        sz_b, seq_len, _ = v.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(-1, d_k)  # (n*b) x d_k

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (n_head, 1)
            )  # replicate pad_mask for each head (nxb) x lk

        # attn   is of shape [B*H*W*h, 1, T], e.g. [2*32*32*16=32768 x 1 x 4], e.g. Size([32768, 1, 4])
        # v      is of shape [B*H*W*h, T, self.d_in/h], e.g. [2*32*32*16=32768 x 4 x 256/16=16], e.g. Size([32768, 4, 16])
        # output is of shape [B*H*W*h, 1, h], e.g. [2*32*32*16=32768 x 1 x 16], e.g. Size([32768, 1, 16])
        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(n_head * sz_b, seq_len, -1)
        if weight_v:
            output, attn = self.attention(q, k, v, pad_mask=pad_mask, return_comp=return_comp, weight_v=weight_v)
            if return_comp:
                output, attn, comp = self.attention(q, k, v, pad_mask=pad_mask, return_comp=return_comp, weight_v=weight_v)
        else:
            attn = self.attention(q, k, v, pad_mask=pad_mask, return_comp=return_comp, weight_v=weight_v)

        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        if weight_v:
            output = output.view(n_head, sz_b, 1, d_in // n_head)
            output = output.squeeze(dim=2)

            # re-shaped attn to   [h x B*H*W x T], e.g. torch.Size([16, 2048, 4])
            #   in utae.py this is torch.Size([h, B, T, 32, 32])
            # re-shaped output to [h x B*H*W x d_in/h], e.g. torch.Size([16, 2048, 16])
            #   in utae.py this is torch.Size([B, 128, 32, 32])

            if return_comp:
                return output, attn, comp
            else:
                return output, attn

        return attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, pad_mask=None, return_comp=False):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        if return_comp:
            comp = attn
        # attn   is of shape [B*H*W*h, 1, T], e.g. [2*32*32*16=32768 x 1 x 4]
        # v      is of shape [B*H*W*h, T, self.d_in/h], e.g. [2*32*32*16=32768 x 4 x 256/16=16]
        # output is of shape [B*H*W*h, 1, h], e.g. [2*32*32*16=32768 x 1 x 16], e.g. Size([32768, 1, 16])
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn

# no longer using dropout (before upsampling)
# but optionally doing attn*v weighting
class ScaledDotProductAttentionSmall(nn.Module):
    """Scaled Dot-Product Attention
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        #self.dropout = nn.Dropout(attn_dropout) # moved dropout after bilinear interpolation
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, pad_mask=None, return_comp=False, weight_v=False):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        if return_comp:
            comp = attn
        # attn   is of shape [B*H*W*h, 1, T], e.g. [2*32*32*16=32768 x 1 x 4]
        # v      is of shape [B*H*W*h, T, self.d_in/h], e.g. [2*32*32*16=32768 x 4 x 256/16=16]
        # output is of shape [B*H*W*h, 1, h], e.g. [2*32*32*16=32768 x 1 x 16], e.g. Size([32768, 1, 16])
        attn = self.softmax(attn)
        
        """
        # no longer using dropout on attention matrices before the upsampling
        # this is now done after bilinear interpolation only

        attn = self.dropout(attn)
        """

        if weight_v:
            # optionally using the weighted values
            output = torch.matmul(attn, v)

            if return_comp:
                return output, attn, comp
            else:
                return output, attn
        return attn