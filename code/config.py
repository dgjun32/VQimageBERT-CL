import easydict
from easydict import EasyDict as edict

cfg = edict()

cfg.model = edict()
cfg.model.h_dim = 768
cfg.model.n_layers = 12
cfg.model.vocab_size = 8192
cfg.model.seq_len = 256
cfg.model.n_heads = 8

cfg.path = edict()
cfg.path.train_image_dir = '../data/train'
cfg.path.val_image_dir = '../data/val'

cfg.train = edict()
cfg.train.gamma = 0.7
cfg.train.optimizer = 'torch.optim.Adam'
cfg.train.lr_sched = 'torch.optim.lr_scheduler.OneCycleLR'
cfg.train.n_epochs = 100
cfg.train.n_steps = 1000000
cfg.train.batch_size = 4
cfg.train.lr = 0.01
cfg.train.betas = (0.9, 0.95)
