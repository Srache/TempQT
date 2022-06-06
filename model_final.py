import torch
import torch.nn as nn
from vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from functools import partial
import torch.nn.functional as F


class IQAModel(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super(IQAModel, self).__init__(*args, **kwargs)
        self.conv_final = nn.Sequential(
            nn.AdaptiveAvgPool2d(32),
            nn.AdaptiveMaxPool2d(32),
            nn.Flatten(),
            nn.Linear(1024, 768),
        )
        self.regression_mos = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(768, 256),
            nn.PReLU(),
            nn.Linear(256, 1),
            # nn.Sigmoid()
        )

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        attn_weights = []
        patch_list = []
        for blk in self.blocks:
            x, weights = blk(x)
            attn_weights.append(weights)
            patch_list.append(x[:, 1:])

        x = self.norm(x)
        return x[:, 0], x[:, 1:], attn_weights, patch_list

    def forward(self, x):
        dist, emap = x
        x_cls, x_patch, attn_weights, patch_list = self.forward_features(dist)
#
#        attn_weights = torch.stack(attn_weights)
#        attn_weights = torch.mean(attn_weights, dim=2)
#        n, _, _ = x_patch.shape
#        am = attn_weights.sum(0)[:, 0, 1:].reshape([n, 14, 14]).unsqueeze(1)
#        # am = am.detach().clone()
#        am = F.interpolate(am, scale_factor=4, mode='bilinear', align_corners=True)
#        am = F.interpolate(am, scale_factor=4, mode='bilinear', align_corners=True)

        emap = self.conv_final(emap)
        # emap = emap * am
        emap = emap + x_cls
        # emap = 0.5 * emap + 0.5 * x_cls
        out = self.regression_mos(emap)
        return out, attn_weights


@register_model
def vit_IQAModel(pretrained=True, **kwargs):
    model = IQAModel(
        patch_size=16, embed_dim=768, depth=12, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        ckpt = torch.hub.load_state_dict_from_url(
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth",
            map_location="cpu", check_hash=True
        )
        model_dict = model.state_dict()

        ignored_keys = ['fc.weight', 'fc.bias']
        for k in ignored_keys:
            if k in ckpt and ckpt[k].shape != model_dict[k].shape:
                print(f'Removing key {k} from pretrained checkpoint')
                del ckpt[k]
        pretrained_dict = {k: v for k, v in ckpt.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


if __name__ == '__main__':
    model = vit_IQAModel()
    print(model.state_dict)
    dist = torch.randn(4, 3, 224, 224)
    dist = (dist, dist)
    logits, _ = model(dist)
    print(logits.shape)
    print(len(_))
    # print(model)


    # inp = torch.randn(1, 3, 16, 16)
    # ups = UpSampling(3, 1)
    # print(ups(inp).shape)
