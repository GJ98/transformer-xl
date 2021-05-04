import torch
from torch import nn


class PositionalEmbedding(nn.Module):

    def __init__(self, max_len: int, d_embed: int, device):
        """positional encoding 구현 클래스

        Args:
            max_len (int): max length(=q_len + m_len)
            d_embed (int): embedding dim
            device: device type
        """

        super().__init__()
        self.pos_enc = torch.zeros(max_len, d_embed, device=device)
        self.pos_enc.requires_grad = False

        pos = torch.arange(start=0, end=max_len, device=device).unsqueeze(1)

        _2i = torch.arange(start=0, end=d_embed, step=2, device=device)

        self.pos_enc[:, 0::2] = torch.sin(pos / 10000 ** (_2i / d_embed))
        self.pos_enc[:, 1::2] = torch.cos(pos / 10000 ** (_2i / d_embed))

    def forward(self, k_len: int):
        """forward 함수
        
        Args:
            k_len: key length
        Returns:
            output(torch.Tensor(k_len, d_embed)): position encoding 결과값
        """

        return self.pos_enc[:k_len, :]
    