import torch
from torch import nn
import math


class ScaledDotProductAttention(nn.Module):

    def __init__(self, device, p: float=None):
        """scaled dot product attention 구현 클래스
        
        Args:
            device: device type
            p (float): dropout probability
        """

        super().__init__()
        self.device = device

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = None if p is None else nn.Dropout(p=p)

    def forward(self, 
                q: torch.Tensor, 
                k: torch.Tensor, 
                v: torch.Tensor,
                _r: torch.Tensor, 
                uT: torch.Tensor, 
                vT: torch.Tensor, 
                mask: torch.Tensor):
        """forward 함수

        Args:
            q (torch.Tensor(bs, head, q_len, d_head)): query
            k (torch.Tensor(bs, head, k_len, d_head)): key
            v (torch.Tensor(bs, head, k_len, d_head)): value
            _r (torch.Tensor(head, m_len + q_len, d_head)): position embedding
            uT (torch.Tensor(n_head, 1, d_head)): segment bias
            vT (torch.Tensor(n_head, 1, d_head)): position bias
            mask (torch.Tenso(bs, 1, q_len, k_len)): masking idx

        Returns
            output (torch.Tensor(bs, head, q_len, d_head)): forward 출력값
        """
                
        d_head = q.size(-1)

        Q_uT = q + uT
        AC = (Q_uT @ k.transpose(-1, -2))

        Q_vT = q + vT
        BD = (Q_vT @ _r.transpose(-1, -2))
        BD = self.left_shift(BD)

        weight = AC + BD
        weight /= math.sqrt(d_head)

        if mask is not None:
            weight.masked_fill(mask==False, 1e-12)

        weight = self.softmax(weight)

        if self.dropout is not None:
            weight = self.dropout(weight)

        output = weight @ v

        return output
        
    def left_shift(self, x: torch.Tensor):
        """left shift 함수

        Args:
            x (torch.Tensor(bs, head, q_len, k_len)): before left shift BD

        Returns:
            x (torch.Tensor(bs, head, q_len, k_len)): left shifted BD

            ex.     (0)              (2)                (3)              (4)             (5)
                a00 a01 a02      0 a00 a01 a02       0  a00 a01      a02  0  a10     a02  0   0
                a10 a11 a12  =>  0 a10 a11 a12  =>  a02  0  a10  =>  a11 a12  0  =>  a11 a12  0
                a20 a21 a22      0 a20 a21 a22      a11 a12  0       a20 a21 a22     a20 a21 a22
                                                    a19 a21 a22
        """

        bs, head, q_len, k_len = x.size()
        m_len = k_len - q_len

        row_pad = torch.zeros(bs, head, q_len, 1).to(self.device)
        col_pad = torch.zeros(bs, head, m_len, k_len + 1).to(self.device)

        x = torch.cat([row_pad, x], dim=-1)
        x = torch.cat([x, col_pad], dim=-2)
        x = x.view(bs, head, -1, k_len)

        x_1 = x[:, :, 1:q_len + 1,:q_len].tril(diagonal=1) 
        x_2 = x[:, :, :q_len, q_len:k_len]
        output = torch.cat([x_2, x_1], dim=-1)

        return output