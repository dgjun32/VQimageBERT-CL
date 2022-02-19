import torch
import torch.nn as nn
from model import VQImageGPT
from config import cfg

class NLLobj(nn.Module):
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


class VQImageGPTModel(nn.Module):
    def __init__(self, VQImageGPT, cfg):
        super().__init__()
        self.vq_imagegpt = VQImageGPT
        self.nll = NLLobj(cfg.model.h_dim, cfg.model.vocab_size)
    
    def forward(self, x):
        return self.nll(*self.vq_imagegpt(x))

