import torch
from torch import nn
from model.attentions.scale_dot_product import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, 
                 head: int, 
                 d_model: int,
                 d_head: int, 
                 device,
                 p: float=None): 
        """multi head attention 구현 클래스

        Args:
            head (int): parallel attention layers
            d_model (int): input, output dim
            d_head (int): key, query, value dim
            device: device type
            p (float): dropout probability
        """

        super().__init__()

        self.head = head
        self.d_head = d_head

        self.W_q = nn.Linear(in_features=d_model, out_features=head * d_head, bias=False) 
        self.W_kE = nn.Linear(in_features=d_model, out_features=head * d_head, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=head * d_head, bias=False)
        self.W_kR = nn.Linear(in_features=d_model, out_features=head * d_head, bias=False)

        self.attention = ScaledDotProductAttention(device=device, p=p)

        self.W_o = nn.Linear(in_features=head * d_head, out_features=d_model, bias=False)


    def forward(self, h, r, uT, vT, m=None, mask=None):
        """forward 함수

        Args:
            h (torch.Tensor(bs, q_len, d_model)): current hidden
            r (torch.Tensor(k_len, d_model)): position embedding
            uT (torch.Tensor(head, 1, d_head)): segment bias
            vT (torch.Tensor(head, 1, d_head)): position bias
            m (torch.Tensor(bs, m_len, d_model)): previous hidden
            mask (torch.Tenso(bs, 1, q_len, k_len)): masking idx

        Returns:
            output (torch.Tensor(bs, q_len, d_model)): forward 출력값
        """
        _h = torch.cat([m, h], 1) \
            if m is not None else h

        bs, q_len, k_len = \
            h.size(0), h.size(1), _h.size(1)

        q = self.W_q(_h[:, -q_len:, :])
        k = self.W_kE(_h)
        v = self.W_v(_h)
        _r = self.W_kR(r)

        q = q.view(bs, q_len, self.head, self.d_head).transpose(1, 2)
        k = k.view(bs, k_len, self.head, self.d_head).transpose(1, 2)
        v = v.view(bs, k_len, self.head, self.d_head).transpose(1, 2)
        _r = _r.view(k_len, self.head, self.d_head).transpose(0, 1)

        output = self.attention(q, k, v, _r, uT, vT, mask)
        output = output.transpose(1, 2).contiguous().view(bs, q_len, self.head * self.d_head)

        output = self.W_o(output)

        return output