"""Criss-cross attention.

Reference:
Huang et al., CCNet: Criss-Cross Attention for Semantic Segmentation. ICCV 2019.

This code is modified from:
https://github.com/speedinghzl/CCNet/blob/master/cc_attention/functions.py
"""

import numpy as np
import torch
from torch import nn


class CrissCrossAttention(nn.Module):
    """Criss-Cross Attention."""

    def __init__(self, n_chns, div_fctr=8):
        """Constructor function.

        Args:
        * n_chns: number of channels of input feature maps
        * div_fctr: division factor for number of channels for query/key computation

        Returns: n/a
        """

        super().__init__()

        n_chns_qk = n_chns // div_fctr
        self.conv_qry = nn.Conv2d(n_chns, n_chns_qk, kernel_size=1)
        self.conv_key = nn.Conv2d(n_chns, n_chns_qk, kernel_size=1)
        self.conv_val = nn.Conv2d(n_chns, n_chns, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros((1)))

    def forward(self, x):
        """Perform the forward pass."""

        def _get_inf_tensor(b, h, w):
            inf_mat = torch.diag(-np.inf * torch.ones(h, device=x.device))
            return torch.repeat_interleave(torch.unsqueeze(inf_mat, dim=0), b * w, dim=0)

        bs, _, h, w = x.shape
        proj_qry = self.conv_qry(x)
        proj_qry_h = proj_qry.permute(0, 3, 2, 1).contiguous().view(bs * w, h, -1)
        proj_qry_w = proj_qry.permute(0, 2, 3, 1).contiguous().view(bs * h, w, -1)
        proj_key = self.conv_key(x)
        proj_key_h = proj_key.permute(0, 3, 1, 2).contiguous().view(bs * w, -1, h)
        proj_key_w = proj_key.permute(0, 2, 1, 3).contiguous().view(bs * h, -1, w)
        proj_val = self.conv_val(x)
        proj_val_h = proj_val.permute(0, 3, 1, 2).contiguous().view(bs * w, -1, h)
        proj_val_w = proj_val.permute(0, 2, 1, 3).contiguous().view(bs * h, -1, w)
        energy_h_raw = torch.bmm(proj_qry_h, proj_key_h) + _get_inf_tensor(bs, h, w)
        energy_h = energy_h_raw.view(bs, w, h, h).permute(0, 2, 1, 3)
        energy_w = torch.bmm(proj_qry_w, proj_key_w).view(bs, h, w, w)
        concate = self.softmax(torch.cat([energy_h, energy_w], dim=3))

        att_h = concate[:, :, :, :h].permute(0, 2, 1, 3).contiguous().view(bs * w, h, h)
        att_w = concate[:, :, :, h:h + w].contiguous().view(bs * h, w, w)
        out_h = torch.bmm(proj_val_h, att_h.permute(0, 2, 1)).view(bs, w, -1, h).permute(0, 2, 3, 1)
        out_w = torch.bmm(proj_val_w, att_w.permute(0, 2, 1)).view(bs, h, -1, w).permute(0, 2, 1, 3)

        return self.gamma * (out_h + out_w) + x


if __name__ == '__main__':
    # unit-test
    model = CrissCrossAttention(64)
    inputs = torch.randn(2, 64, 5, 6)
    outputs = model(inputs)
    print('inputs.shape: {}'.format(inputs.shape))
    print('outputs.shape: {}'.format(outputs.shape))
    assert inputs.shape == outputs.shape, 'inconsistent input / output tensor shape'
