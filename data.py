import os

import torch
import torchtext
from torchtext.legacy.data import Field, BucketIterator, ReversibleField
from torchtext.data.utils import get_tokenizer
from torchtext.legacy.datasets.translation import Multi30k 
from torchtext.legacy.datasets import WikiText103

tgt_len = 128
batch_size = 16
eval_batch_size = 16
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def batchify(data, bsz):
    data = TEXT.numericalize([data.examples[0].text])
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).contiguous()
    return data.to(device)

if not os.path.exists("./.data/cache.pt"):
    TEXT = Field(tokenize=get_tokenizer("basic_english"),
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)
 
    train_txt, val_txt, test_txt = WikiText103.splits(TEXT)

    TEXT.build_vocab(train_txt)

    train_data = batchify(train_txt, batch_size)
    val_data = batchify(val_txt, eval_batch_size)
    test_data = batchify(test_txt, eval_batch_size)

    itos = TEXT.vocab.itos
    stoi = TEXT.vocab.stoi

    data = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "itos": TEXT.vocab.itos,
        "stoI": TEXT.vocab.stoi
    }

    torch.save(data, "./.data/cache.pt")

else:
    data = torch.load("./.data/cache.pt")
    train_data = data["train"]
    val_data = data["val"]
    test_data = data["test"]
    itos = data["itos"]
    stoi = data["stoI"]
    
vocab_size = len(itos)
#torch.Size([60, 1722427])
cut_off = train_data.size(1) // tgt_len

src = train_data.narrow(1, 1, cut_off * tgt_len)
src = torch.chunk(src, cut_off, dim=1)

tgt = train_data.narrow(1, 0, cut_off * tgt_len)
tgt = torch.chunk(tgt, cut_off, dim=1)

train_iter = zip(src, tgt)


print(vocab_size)
'''
for src, tgt in train_iter:
    print(src.size())
    print(tgt.size())
'''
'''
s = []
t = []
for src, tgt in train_iter:
    a = src[:100]
    b = tgt[:100]
    for i in range(100):
        s.append(itos[a[i]])
        t.append(itos[b[i]])
    print(" ".join(s))
    print(" ".join(t))
    break
'''
