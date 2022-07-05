import torch.nn as nn

class noname(nn.Module):

    def __init__(self, config):

        # EPI => 按行拆分 625 * 9
        self.embed_dim = config.epi_w * config.epi_h
        self.dim = config.epi_fe_dim

        self.to_embed = nn.Sequential(
            Rearrange('b h w -> b (h w)'),
            nn.Layernorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.dim)
        )

        self.net = nn.Sequential(
            nn.Linear(self.dim, self.d_ff),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.dim, 1),
        )

        self.


    def forward(self, x):
        # x (bs, h * w) => (bs, dim)
        out = to_embed(x)
        # return 1 output
        return net(out)


class get_epi():

