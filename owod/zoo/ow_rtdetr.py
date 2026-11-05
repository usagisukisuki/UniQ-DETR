"""RT-DETR wrapper that also exposes the backbone attention map used by OW-DETR / CAT pseudo-labelling.

OW-DETR scores unmatched query boxes by the channel-mean activation of the raw backbone C4 feature
(ResNet layer3, 1024 ch, stride 16), taken before the encoder's input projection (docs/baselines/OWDETR_SPEC.md).
"""
import torch.nn as nn

from src.core import register

__all__ = ['OWRTDETR']


@register()
class OWRTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module,
                 attn_map_level=None):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.attn_map_level = attn_map_level          # index into the backbone outputs (1 = C4), None = off

    def forward(self, x, targets=None):
        feats = self.backbone(x)
        out = self.decoder(self.encoder(feats), targets)
        if self.attn_map_level is not None and self.training:
            out['attn_map'] = feats[self.attn_map_level].detach().float().mean(1)     # [B, h, w]
        return out

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
