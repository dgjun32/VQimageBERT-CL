import torch
import torch.nn as nn

from model import VQImageBERT
from objective import VQImageBERTObj
from datasets import ImageDataset
from config import cfg
from trainer import Trainer

if __name__ == '__main__':
    model_obj = VQImageBERTObj(cfg)
    train_dataset = ImageDataset('train', cfg)
    val_dataset = ImageDataset('val', cfg)

    algo = Trainer(train_dataset, val_dataset, model_obj, cfg)
    algo.train(start_step = 0)