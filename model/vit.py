"""
normal-VisionTransformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def get_attn_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    attn_mask = seq_k.data.eq(i_pad)
    attn_mask = attn_mask.unsqueeze(1).expand(batch_size, len_q, len_k)
    return attn_mask


class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv_enc = nn.Conv2d(2048, self.config.d_hidn, kernel_size=1)

        self.transformer = VisionTransformer(self.config)

        # MLP head
        self.projection = nn.Sequential(
            nn.Linear(self.config.d_hidn, self.config.d_MLP_head, bias=False),
            nn.GELU(),
            nn.Linear(self.config.d_MLP_head, self.config.n_output, bias=False)
        )

    def forward(self, mask_inputs, feat_dis_org):
        feat_dis_org = self.conv_enc(feat_dis_org)

        # （batch_size, n_enc_seq+1, d_hidn) -> (batch_size, d_hidn)
        enc_outputs = self.transformer(mask_inputs, feat_dis_org)
        # print("enc_outputs", enc_outputs.size())
        enc_outputs = enc_outputs[:, 0, :]

        # (batch_size, n_output)
        pred = self.projection(enc_outputs)

        return pred


class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(self.config)

    def forward(self, mask_inputs, feat_dis_org_embed):
        enc_outputs, enc_self_attn_probs = self.encoder(mask_inputs, feat_dis_org_embed)

        return enc_outputs


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.scale_org_embedding = nn.Parameter(torch.rand(1, self.config.d_hidn, 1, 1))

        self.pos_embedding = nn.Parameter(torch.randn(1, self.config.Grid, self.config.Grid, self.config.d_hidn))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.config.d_hidn))
        self.dropout = nn.Dropout(self.config.emb_dropout)

        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.n_layer)])

    def forward(self, mask_inputs, feat_dis_org_embed):
        # feat_dis_org_embed : batch_size x (C = 384) x (h = H / patch_size) x (w = W / patch_size)

        # learnable scale embedding
        scale_org_embed = repeat(self.scale_org_embedding, '() c () () -> b c h w', b=self.config.batch_size, h=1, w=20)
        feat_dis_org_embed += scale_org_embed

        # learnable 2D spatial embedding
        # original scale
        b, c, h, w = feat_dis_org_embed.size()
        spatial_org_embed = torch.zeros(1, self.config.d_hidn, h, w).to(self.config.device)
        for i in range(h):
            for j in range(w):
                t_i = int((i / h) * self.config.Grid)
                t_j = int((j / w) * self.config.Grid)
                spatial_org_embed[:, :, i, j] = self.pos_embedding[:, t_i, t_j, :]
        spatial_org_embed = repeat(spatial_org_embed, '() c h w -> b c h w', b=self.config.batch_size)
        feat_dis_org_embed += spatial_org_embed

        b, c, h, w = feat_dis_org_embed.size()
        feat_dis_org_embed = torch.reshape(feat_dis_org_embed, (b, c, h * w))
        feat_dis_org_embed = feat_dis_org_embed.permute((0, 2, 1))

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=self.config.batch_size)
        # print("cls_tokens =>", cls_tokens.size())
        x = torch.cat((cls_tokens, feat_dis_org_embed), dim=1)
        # print("x =>", x.size())

        outputs = self.dropout(x)

        #  (batch_size, n_enc_seq+1, n_enc_seq+1)
        attn_mask = get_attn_mask(mask_inputs, mask_inputs, self.config.i_pad)

        attn_probs = []
        for layer in self.layers:
            # (batch_size, n_enc_seq+1, d_hidn), (batch_size, attn_head, n_enc_seq+1, n_enc_seq+1)
            outputs, attn_prob = layer(outputs, attn_mask)
            attn_probs.append(attn_prob)

        return outputs, attn_probs


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.self_attn = MutliHeadAttention(self.config)
        self.ln1 = nn.LayerNorm(self.config.d_hidn, eps=self.config.ln_eps)
        self.pos_attn = PoswiseFeedForwardNet(self.config)
        self.ln2 = nn.LayerNorm(self.config.d_hidn, eps=self.config.ln_eps)

    def forward(self, inputs, attn_mask):
        # (batch_size, n_enc_seq, d_hidn) (batch_size, attn_head, n_enc_seq, n_enc_seq)
        att_outputs, attn_prob = self.self_attn(inputs, inputs, inputs, attn_mask)
        att_outputs = self.ln1(att_outputs)

        # (batch_size, n_enc_seq, d_hidn)
        ffn_outputs = self.pos_attn(inputs)
        ffn_outputs = self.ln2(ffn_outputs + att_outputs)

        return ffn_outputs, attn_prob


class MutliHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.W_Q = nn.Linear(self.config.d_hidn, self.config.attn_head * self.config.d_head)
        self.W_K = nn.Linear(self.config.d_hidn, self.config.attn_head * self.config.d_head)
        self.W_V = nn.Linear(self.config.d_hidn, self.config.attn_head * self.config.d_head)

        self.scale_dot_attn = ScaledDotProductAttention(self.config)
        self.linear = nn.Linear(self.config.d_head * self.config.attn_head, self.config.d_hidn)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)

        # (batch_size, attn_head, n_q_seq, d_head)
        q_s = self.W_Q(Q).view(batch_size, -1, self.config.attn_head, self.config.d_head).transpose(1, 2)
        # (batch_size, attn_head, n_q_seq, d_head)
        k_s = self.W_K(K).view(batch_size, -1, self.config.attn_head, self.config.d_head).transpose(1, 2)
        # (batch_size, attn_head, n_q_seq, d_head)
        v_s = self.W_V(V).view(batch_size, -1, self.config.attn_head, self.config.d_head).transpose(1, 2)

        # (batch_size, attn_head, n_q_seq, n_k_seq)
        # print("before => attn_mask", attn_mask.size())
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config.attn_head, 1, 1)
        # print("after => attn_mask", attn_mask.size())

        # (batch_size, attn_head, n_q_seq, d_head), (batch_size, attn_head, n_q_seq, n_k_seq)
        context, attn_prob = self.scale_dot_attn(q_s, k_s, v_s, attn_mask)

        # (batch_size, attn_head, n_q_seq, attn_head * d_head)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.config.attn_head * self.config.d_head)

        # (batch_size, attn_head, n_q_seq, embd)
        output = self.linear(context)
        output = self.dropout(output)

        return output, attn_prob


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv1d(self.config.d_hidn, self.config.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(self.config.d_ff, self.config.d_hidn, kernel_size=1)
        self.gelu = F.gelu
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x):
        # (batch_size, d_ff, n_seq)
        out = self.conv1(x.transpose(1, 2))
        out = self.gelu(out)

        # 6.16 新加入
        out = self.dropout(out)
        # (batch_size, n_seq, d_hidn)
        out = self.conv2(out).transpose(1, 2)
        out = self.dropout(out)

        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1 / (self.config.d_head ** 0.5)

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2))
        scores = scores.mul_(self.scale)
        scores.masked_fill_(attn_mask, -1e9)

        # (batch_size, attn_head, n_q_seq, d_v)
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(attn_prob)

        # (batch_size, attn_head, n_q_seq, d_v)
        context = torch.matmul(attn_prob, V)

        return context, attn_prob
