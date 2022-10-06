import torch
import torch.nn.functional as F
from torch import nn
from model.vit2 import ViT2

from einops import rearrange, repeat

def exists(val):
    return val is not None

class DistillWrapper(nn.Module):
    def __init__(self, config, teacher_logits, student_logits):
        super().__init__()
        self.config = config

        self.teacher_logits = teacher_logits
        self.student_logits = student_logits

        dim = config.d_hidn
        self.temperature = config.temperature
        self.alpha = self.config.alpha
        self.hard = self.config.hard

        self.distillation_token = nn.Parameter(torch.randn(1, 1, dim))
        # TODO
        self.distill_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.config.n_output)
        )

    def forward(self, img, labels, temperature=None, alpha=None):
        b, *_ = img.shape
        alpha = alpha if exists(alpha)  else self.alpha
        T = temperature if exists(temperature)  else self.temperature


        students_logits, distill_tokens = self.student(img, distill_token=self.distillation_token)
        distill_logits = self.distill_mlp(distill_tokens)

        loss = F.cross_entropy(students_logits, labels)

        if not self.hard:
            distill_loss = F.kl_div(
                F.log_softmax(distill_logits / T, dim=-1),
                F.softmax(self.teacher_logits / T, dim=-1).detach(),
            reduction='batchmean'
            )
            distill_loss *= T ** 2

        else:
            teacher_labels = self.teacher_logits.argmax(dim=-1)
            distill_loss = F.cross_entropy(distill_logits, teacher_labels)

        return loss * (1 - alpha) + distill_loss * alpha


class DistillMixin:
    def forward(self, img, distill_token=None):
        distilling = exists(distill_token)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        if distilling:
            distill_tokens = repeat(distill_token, '() n d -> b n d', b=b)
            x = torch.cat((x, distill_tokens), dim=1)

        x = self._attend(x)

        if distilling:
            x, distill_tokens = x[:, :-1], x[:, -1]

        x = x[:, 0]

        out = self.mlp_head(x)

        if distilling:
            return out, distill_tokens

        return out
    
class DistillableViT(DistillMixin, ViT2):
    def __init__(self, config):
        self.config = config
        self.dim = self.config.d_hidn
        self.n_class = self.config.n_output
        
    def to_vit(self):
        v = ViT2(self.config)
        v.load_state_dict(self.state_dict())
        return v

    def _attend(self, x):
        x = self.dropout(x)
        x = self.transformer(x)
        return x