import torch
import torch.nn as nn
from model import VQImageGPT
from config import cfg

class GenerationObj(nn.Module):
    def __init__(self, h_dim, vocab_size):
        super().__init__()
        # compute likelihood 
        self.linear = nn.Linear(h_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.nllloss = nn.NLLLoss()

    def forward(self, x, target):
        '''
        x : torch.FloatTensor shape of [batch_size x seq_len x h_dim]
        target : torch.LongTensor shape of [batch_size x seq_len]
        '''
        log_probs = self.softmax(self.linear(x))
        loss = self.nllloss(log_probs.permute(0,2,1), target)
        return loss


class MIMObj(nn.Module):
    def __init__(self, h_dim, vocab_size):
        super().__init__()
        self.linear = nn.Linear(h_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.nllloss = nn.NLLLoss()
    
    def forward(self, x, target):
        log_probs = self.softmax(self.linear(x))
        mlm_loss = self.nllloss(log_probs.permute(0,2,1), target)
        return mlm_loss


class NCEobj(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        pass


class VQImageBERTObj(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vq_imagegpt = VQImageGPT(**cfg.model)
        self.mim_loss = MIMObj(cfg.model.h_dim, cfg.model.vocab_size)
    
    def forward(self, x):
        return self.mim_loss(*self.vq_imagegpt(x))

