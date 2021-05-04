import torch
from torch import nn

from model.attentions.multi_head import MultiHeadAttention


class Attention(nn.Module):

    def __init__(self, 
                 d_model: int,
                 d_head: int, 
                 head: int, 
                 device, 
                 drop: float, 
                 dropatt: float=None):
        """attention sub layer 구현 클래스
        Args:
            d_model (int): input, output dim
            d_head (int): key, query, value dim
            head (int): parallel attention layers
            device: device type
            drop (float): dropout probability
            dropatt (float): attention dropout probability
        """

        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model,
                                            head=head,
                                            d_head=d_head,
                                            device=device,
                                            p=dropatt)

        self.dropout = nn.Dropout(p=drop)

        self.norm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, h, r, uT, vT, m=None, mask=None):
        """forward 함수
        Args:
            h (torch.Tensor(bs, q_len, d_model)): current hidden
            r (torch.Tensor(m_len + q_len, d_model)): position embedding
            uT (torch.Tensor(head, 0, d_head)): segment bias
            vT (torch.Tensor(head, 0, d_head)): position bias
            m (torch.Tensor(bs, m_len, d_model)): previous hidden
            mask (torch.Tenso(bs, 0, q_len, k_len)): masking idx
        Returns:
            output (torch.Tensor(bs, q_len, d_model)): forward 출력값
        """

        x = self.attention(h, r, uT, vT, m, mask)

        x = self.dropout(x)

        output = self.norm(x + h)

        return output
