  
import torch
from torch import nn

from model.sub_layers.attention import Attention
from model.sub_layers.feed_forward import FeedForward


class TransformerXLBlock(nn.Module):

    def __init__(self, 
                 d_model: int, 
                 d_head: int, 
                 head: int, 
                 d_ff: int, 
                 device,
                 drop: float,
                 dropatt: float):
        """decoder layer 구현 클래스
        Args:
            d_model (int): intput, output dim
            d_head (int): key, query, value dim
            head (int): parallel attention layers
            d_ff (int): hidden dim
            device: device type
            drop (float): dropout probability
            dropatt (float): attention dropout probability
        """

        super().__init__()
        self.attention = Attention(d_model=d_model,
                                   d_head=d_head,
                                   head=head,
                                   device=device,
                                   drop=drop,
                                   dropatt=dropatt)

        self.feed_forward = FeedForward(d_model=d_model,
                                        d_ff=d_ff,
                                        p=drop)

    def forward(self, h, r, uT, vT, m=None, mask=None):
        """forward 함수
        Args:
            h (torch.Tensor(bs, q_len, d_model)): current hidden
            r (torch.Tensor(k_len, d_model)): position embedding
            uT (torch.Tensor(head, 1, d_head)): segment bias
            vT (torch.Tensor(head, 1, d_head)): position bias
            m (torch.Tensor(bs, m_len, d_model)): previous hidden
            mask (torch.Tenso(bs, 0, q_len, k_len)): masking idx
        Returns:
            output (torch.Tensor(bs, q_len, d_model)): forward 출력값
        """
     
        x = self.attention(h, r, uT, vT, m, mask)

        output = self.feed_forward(x)

        return h