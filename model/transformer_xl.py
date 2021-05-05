import torch
from torch import nn

from model.blocks.transformer_xl import TransformerXLBlock
from model.embeds.position import PositionalEmbedding


class Transformer_XL(nn.Module):

    def __init__(self, 
                 vocab_size: int,
                 max_len: int,
                 m_len: int,
                 d_model: int, 
                 d_head: int,
                 head: int, 
                 d_ff: int, 
                 n_layer: int, 
                 device,
                 drop: float,
                 dropatt: float=None):
        """transformer-XL 구현 클래스
        Args:
            vocab_size (int): vocabulary size
            max_len (int): max length
            m_len (int): memeory length
            d_model (int): intput, output dim
            d_head (int): key, query, value dim
            head (int): parallel attention layers
            d_ff (int): hidden dim
            n_layer (int): number of layer
            drop (float): dropout probability
            dropatt (float): attention dropdout probability
            device: device type
        """

        super().__init__()
        self.device = device
        self.pad = 0
        self.m_len = m_len

        self.dropout = nn.Dropout(p=drop)
        self.embed_layer = nn.Embedding(num_embeddings=vocab_size,
                                        embedding_dim=d_model)

        self.pos_emb = PositionalEmbedding(max_len=max_len,
                                           d_embed=d_model,
                                           device=device)

        self.uT = nn.Parameter(torch.Tensor(head, 1, d_head).to(device))
        self.vT = nn.Parameter(torch.Tensor(head, 1, d_head).to(device))

        self.layers = nn.ModuleList([TransformerXLBlock(d_model=d_model,
                                                        d_head=d_head,
                                                        head=head,
                                                        d_ff=d_ff,
                                                        device=device,
                                                        drop=drop,
                                                        dropatt=dropatt) for _ in range(n_layer)])
        
        self.linear = nn.Linear(in_features=d_model,
                                out_features=vocab_size)

    def forward(self, sequence: torch.Tensor, ms: torch.Tensor=None):
        """forward 함수
        Args:
            sequence (torch.Tensor(bs, q_len)): foward 입력값
            ms (torch.Tensor(n_layer, bs, m_len, d_model)): memory

        Returns:
            output (torch.Tensor(bs, q_len, d_model)): forward 출력값
        """ 

        m_len = ms[0].size(1) if ms is not None else 0
        q_len = sequence.size(1)
        k_len = m_len + q_len

        """
        mask = self.get_pad_mask(q_len, k_len) * \
            self.get_no_peak_mask(q_len, k_len)
        """

        mask = self.get_no_peak_mask(q_len, k_len)
        hs = []

        h = self.embed_layer(sequence)
        h = self.dropout(h)

        r = self.pos_emb(k_len)
        r = self.dropout(r)

        hs.append(h)
        for i, layer in enumerate(self.layers):
            m = None if ms is None else ms[i]
            h = layer(h, r, self.uT, self.vT, m, mask)
            hs.append(h)

        output = self.dropout(h)

        output = self.linear(output)

        new_ms = self.update_mems(hs, ms)

        return output, new_ms

    '''
    def get_pad_mask(self, q_len: int, k_len: int):
        """pad mask 수행 함수
        Args:
            q_len (int): query length
            k_len (int): key length
        Returns:
            mask (torch.Tensor(bs, 1, len_q, len_k)): pad mask
        """

        # batch size x 1 x 1 x len_k
        k = k.ne(self.pad).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1, 1, len_q, 1)
        # batch size x 1 x len_q x 1
        q = q.ne(self.pad).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1, 1, 1, len_k)

        mask = k & q

        return mask
    '''

    def get_no_peak_mask(self, q_len: int, k_len: int):
        """no peak mask 수행 함수
        Args:
            q_len (int): query length
            k_len (int): key length
        Returns:
            mask (torch.Tensor(len_q, len_k)): no peak mask
        """

        m_len = k_len - q_len

        mask = torch.tril(torch.ones(q_len, k_len), m_len)
        mask = mask.type(torch.BoolTensor).to(self.device)

        return mask

    def update_mems(self, hs, ms):
        """memory update 함수

        Args:
            hs (torch.Tensor(n_layer, bs, q_len, d_model)): hidden state
            ms (torch.Tensor(n_layer, bs, m_len, d_model)): previous memory

        Returns:
            new_ms (torch.Tensor(n_layer, bs, m_len, d_model)): current memory
        """

        q_len = hs.size(-2)

        with torch.no_grad():
            new_ms = []

            if ms is None:
                for i in range(len(hs)):
                    new_ms.append(hs[i][:][-self.m_len:].detach())

            else:
                for i in range(len(hs)):
                    cat = torch.cat([ms[i], hs[i]], dim=1)
                    new_ms.append(cat.narrow(1, q_len, self.m_len).detach())

        return new_ms
