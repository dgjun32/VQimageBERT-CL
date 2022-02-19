import typing
import torch
import torch.nn as nn
import torch.nn.functional as F

from vae_model import VQVAE

def future_mask(seq_len: int) -> torch.BoolTensor:
    mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
    mask = mask.triu(1)
    return mask

class PositionwiseFFN(nn.Module):
    def __init__(self,
                h_dim,
                rate = 4,
                dropout = 0.1):
        super().__init__()
        self.mlp_1 = nn.Linear(h_dim, h_dim*rate)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout()
        self.mlp_2 = nn.Linear(h_dim*rate, h_dim)
    
    def forward(self, x):
        x = self.mlp_1(x)
        x = self.sigmoid(x)*x
        x = self.dropout(x)
        x = self.mlp_2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self,
                seq_len,
                n_heads,
                h_dim,
                dropout = 0.1):
        super().__init__()
        self.future_mask = future_mask(seq_len-1)
        self.ln_1 = nn.LayerNorm((seq_len-1, h_dim))
        self.msa = nn.MultiheadAttention(h_dim, n_heads, batch_first=True, dropout=dropout)
        self.ln_2 = nn.LayerNorm((seq_len-1, h_dim))
        self.mlp = PositionwiseFFN(h_dim, rate=4)
    
    def forward(self, x):
        x1 = self.ln_1(x)
        x1= self.msa(x1, x1, x1,
                    attn_mask=self.future_mask,
                    need_weights=False)[0] + x
        x2 = self.ln_2(x1)
        x2 = self.mlp(x2) + x1
        return x2

class VQImageGPT(nn.Module):
    def __init__(self,
                vocab_size,
                seq_len,
                n_layers, 
                n_heads, 
                h_dim, 
                res_h_dim, 
                n_res_layers,
                dropout=0.1):
        super().__init__()
        # pretrained VQ_VAE
        self.vqvae = VQVAE(h_dim, res_h_dim, n_res_layers, vocab_size, h_dim, train=False)
        ##self.vqvae.load_state_dict(torch.load('...'))
        for param in self.vqvae.parameters():
            param.trainable = False

        # positional encoding
        self.position_encoding = nn.parameter.Parameter(torch.randn(seq_len-1, h_dim),
                                                        requires_grad=True)
        # transformer blocks
        self.transformers = nn.Sequential(*[TransformerBlock(seq_len, n_heads, h_dim, dropout)
                                             for _ in range(n_layers)])

    def forward(self, x):
        z_q, image_tokens = self.vqvae(x)
        z_q = z_q.view(z_q.shape[0], z_q.shape[1], z_q.shape[2]*z_q.shape[3]).permute(0,2,1)[:,:-1,:]
        x = z_q + self.position_encoding
        x = self.transformers(x)
        target = image_tokens.squeeze(1).view(image_tokens.shape[0],
                                            image_tokens.shape[2]*image_tokens.shape[3])[:,1:]
        return x, target

        

