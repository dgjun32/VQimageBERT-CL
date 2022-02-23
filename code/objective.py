import torch
import torch.nn as nn
from model import VQImageBERT
from config import cfg


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


class NCEObj(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, embedding, masked_target):
        '''
        x : last hidden states shape of (batch_size, seq_len, h_dim)
        embedding : ground truth token embedding shape of (batch_size, seq_len, h_dim)
        masked_target : masked token id shape of (batch_size, seq_len)
        '''
        mask_idx = torch.where(masked_target==8192)
        loss, count = 0.0, 0.0
        for i, j in zip(*mask_idx):
            i, j = i.item(), j.item()
            f, m = embedding[i,j,:], x[i,j,:]
            numer = torch.dot(f, m).exp()
            denumer = torch.matmul(f.view(1,-1), x.view(-1,1024).permute(1,0)).exp().sum()
            loss += torch.log(numer/denumer)
            count += 1
        return -(loss / count)


class VQImageBERTObj(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vq_imagebert = VQImageBERT(**cfg.model)
        self.mim_loss = MIMObj(cfg.model.h_dim, cfg.model.vocab_size)
        self.nce_loss = NCEObj()
    
    def forward(self, x):
        hidden_states, target, embedding, masked_target = self.vq_imagebert(x)
        mim_loss = self.mim_loss(hidden_states, target)
        nce_loss = self.nce_loss(hidden_states, embedding, masked_target)
        return mim_loss, nce_loss

