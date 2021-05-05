import os
import platform
import torch

#device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#path
root_dir = os.path.abspath(os.curdir)
_ = '\\' if platform.system() == 'Windows' else '/'
if root_dir[len(root_dir) - 1] != _: root_dir += _
model_dir = root_dir + "saved{_}".format(_=_)
result_dir = root_dir + "result{_}".format(_=_)

#model
batch_size = 60
d_model = 410
d_head = 41
head = 10
d_ff = 2100
n_layer = 6
max_len = 300
m_len = 150
tgt_len = 150
drop = 0.1
dropatt = 0.0

#optimizer
epoch = 1000
init_lr = 1e-5
adam_eps = 5e-9
weight_decay = 5e-4
factor = 0.9
patience = 10
inf = float('inf')
warmup = 100