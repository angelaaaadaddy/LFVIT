import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# classes

class PreNorm(nn.Module):
    def __init__(self, config, fn):
        super().__init__()
        self.config = config
        self.norm = nn.LayerNorm(self.config.d_hidn)
        self.fn = fn
    def forward(self, x):
        return self.fn(self.norm(x))

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(self.config.d_hidn, self.config.d_ff),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_ff, self.config.d_hidn),
            nn.Dropout(self.config.dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = self.config.d_hidn
        self.heads = self.config.attn_head
        self.dim_head = self.config.d_head
        inner_dim = self.dim_head * self.heads

        self.scale = self.dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(self.config.dropout)

        self.to_qkv = nn.Linear(self.dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, self.dim),
            nn.Dropout(self.config.dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([])
        for _ in range(self.config.n_layer):
            self.layers.append(nn.ModuleList([
                PreNorm(self.config.d_hidn, Attention(self.config)),
                PreNorm(self.config.d_hidn, FeedForward(self.config))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT2(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        num_patches = (self.config.image_height // self.config.patch_size) * (self.config.image_width // self.config.patch_size)
        patch_dim = 3 * self.config.patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.config.patch_size, p2=self.config.patch_size),
            nn.Linear(patch_dim, self.config.d_hidn),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, self.config.d_hidn))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.config.d_hidn))
        self.dropout = nn.Dropout(self.config.emb_dropout)

        self.transformer = Transformer(self.config)

        self.mlp_head = nn.Sequential(
            nn.Linear(self.config.d_hidn, self.config.d_MLP_head, bias=False),
            nn.GELU(),
            nn.Linear(self.config.d_MLP_head, self.config.n_output, bias=False)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x[:, 0, :]

        return self.mlp_head(x)
