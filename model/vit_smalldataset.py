import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange



"""
Attention for small dataset
"""
class LSA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.d_hidn
        self.attn_head = config.attn_head
        self.inner_dim = self.dim * self.attn_head
        self.temperature = nn.Parameter(torch.log(torch.tensor(self.dim ** -0.5)))
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(config.dropout)

        self.to_qkv = nn.Linear(self.dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, self.dim),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t:rearrange(t, 'b n (h d) -> b h n d', h = self.attn_head), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        mask = torch.eye(dots.shape[-1], device=dots.device, dtype=torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class SPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_dim = config.patch_size * config.patch_size * 15

        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=config.patch_size, p2=config.patch_size),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, config.d_hidn)
        )

    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        # print("shifted_x", shifted_x)
        x_with_shifts = torch.cat((x, *shifted_x), dim=1)
        # print("x_with_shifts", x_with_shifts.size())
        return self.to_patch_tokens(x_with_shifts)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = self.config.d_hidn
        self.d_ff = self.config.d_ff
        self.net = nn.Sequential(
            nn.Linear(self.dim, self.d_ff),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.d_ff, self.dim),
            nn.Dropout(self.config.dropout)
        )

    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.self_attn = LSA(self.config)
        self.ln1 = nn.LayerNorm(self.config.d_hidn, eps=self.config.ln_eps)
        self.ffn = FeedForward(self.config)
        self.ln2 = nn.LayerNorm(self.config.d_hidn, eps=self.config.ln_eps)

    def forward(self, x):
        attn_out = self.self_attn(x)
        attn_out = self.ln1(x + attn_out)
        ffn_out = self.ffn(attn_out)
        ffn_out = self.ln2(attn_out + ffn_out)

        return ffn_out

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_patches = self.config.n_enc_seq
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.config.d_hidn))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.config.d_hidn))
        self.dropout = nn.Dropout(self.config.emb_dropout)

        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(config.n_layer)])

    def forward(self, embed):

        # b, c, h, w = embed.size()
        # embed = torch.reshape(embed, (b, c, h * w))
        # embed = embed.permute((0, 2, 1))
        # print("embed", embed.size())
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=self.config.batch_size)
        x = torch.cat((cls_tokens, embed), dim=1)
        x += self.pos_embedding
        out = self.dropout(x)

        for layer in self.layers:
            out = layer(out)

        return out

class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(self.config)

        self.to_patch_embedding = SPT(self.config)

        # MLP head
        self.projection = nn.Sequential(
            nn.Linear(self.config.d_hidn, self.config.d_MLP_head, bias=False),
            nn.GELU(),
            nn.Linear(self.config.d_MLP_head, self.config.n_output, bias=False)
        )

    def forward(self, img):
        # print("img", img.size())
        x = self.to_patch_embedding(img)

        enc_out = self.encoder(x)
        enc_out = enc_out[:, 0, :]

        pred = self.projection(enc_out)
        # print("pred", pred.size())

        return pred
