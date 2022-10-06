import torch.nn as nn
from einops.layers.torch import Rearrange

class EPI_special(nn.Module):

    def __init__(self, config):
        super().__init__()
        # EPI => 按行拆分 625 * 9
        self.epi_dim = config.epi_w * config.epi_h
        self.dim = config.d_hidn

        # 625 * 9 => 625 * 434
        self.to_embed = nn.Sequential(
            Rearrange('b h w c -> b (h w) c'),
            nn.Layernorm(self.epi_dim),
            nn.Linear(self.epi_dim, 625)
        )

        # 625 => 313
        self.conv1 = nn.Conv2d(3, 128, kernel_size=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        # 313 => 157
        self.conv2 = nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        # 157 => 79
        self.conv3 = nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        # 79 => 40
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=1, stride=2, bias=False)
        self.bn3 = nn.BatchNorm2d(1024)
        # 40 => 20
        self.conv4 = nn.Conv2d(1024, 2048, kernel_size=1, stride=2, bias=False)
        self.bn4 = nn.BatchNorm2d(2048)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        # x (bs, h * w) => (bs, dim)
        out = self.to_embed(x)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)

        return out


