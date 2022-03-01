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
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.nllloss = nn.NLLLoss()

    def forward(self, x, embedding, masked_target):
        '''
        x : last hidden states shape of (batch_size, seq_len, h_dim)
        embedding : ground truth token embedding shape of (batch_size, seq_len, h_dim)
        masked_target : masked token id shape of (batch_size, seq_len)
        '''
        batch_size, seq_len, h_dim = x.shape
        
        masked_target = masked_target.reshape(batch_size*seq_len) 
        mask_idx = torch.where(masked_target==8192)[0] # shape of (n_masked_token)
        e = embedding.reshape(-1, h_dim)[mask_idx] # shape of (n_masked_token, h_dim)
        h = x.reshape(-1, 768)
        sim = torch.matmul(e, h.T) # shape of (n_masked_token, seq_len*batch_size)
        sim = self.logsoftmax(sim)
        nce_loss = self.nllloss(sim, mask_idx)
        return nce_loss


class VQImageBERTObj(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vq_imagebert = VQImageBERT(**cfg.model)
        self.mim_loss = MIMObj(cfg.model.h_dim, cfg.model.vocab_size)
        self.nce_loss = NCEObj()
    
    def forward(self, x):
        hidden_states, target, embedding, masked_target = self.vq_imagebert(x.cuda())
        mim_loss = self.mim_loss(hidden_states, target)
        nce_loss = self.nce_loss(hidden_states, embedding, masked_target)
        return mim_loss, nce_loss

