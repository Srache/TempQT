import torch
import torch.nn as nn
from vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from functools import partial


class UpSampling(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 features=[1024, 512],
                 conv_out=False):
        super(UpSampling, self).__init__()
        self.net = nn.Sequential(
            self._make_layer(in_channels, features[0]),
            self._make_layer(features[0], features[1]),
            nn.Conv2d(features[1], out_channels, 3, 1, 1) if conv_out else nn.Identity()
        )

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        return self.net(x)


class IQAModel(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super(IQAModel, self).__init__(*args, **kwargs)

        self.ups_list = nn.ModuleList([UpSampling(self.embed_dim, 1, conv_out=False) for _ in range(4)])
        self.ups_final = UpSampling(2048, 1, features=[1024, 256], conv_out=True)

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
        x_cls, x_patch, attn_weights, patch_list = self.forward_features(x)

        # patch reshape
        patches = [patch_list[0], patch_list[2], patch_list[6], patch_list[11]]
        patches = torch.stack(patches)

        patches_ups = []
        N, num_patches, dim = patches[0].shape

        # aggregation
        for idx, p in enumerate(patches):
            if idx == patches.shape[0]-1:
                break
            patches[idx+1] = patches[idx] + patches[idx+1]

        # upsampling
        for idx, p in enumerate(patches):
            p = torch.reshape(p, (N, int(num_patches ** 0.5), int(num_patches ** 0.5), dim))
            p = p.permute(0, 3, 1, 2)
            p = p.contiguous()
            p = self.ups_list[idx](p)
            patches_ups.append(p)

        patchs_cat_4x = torch.cat([patches_ups[0], patches_ups[1], patches_ups[2],
                                   patches_ups[3]], dim=1)
        patchs_cat_16x = self.ups_final(patchs_cat_4x)

        return patchs_cat_16x, attn_weights


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
    dist = torch.randn(4, 3, 224, 224)
    logits, _ = model(dist)
    print(logits.shape)
    print(len(_))
    # print(model)


    # inp = torch.randn(1, 3, 16, 16)
    # ups = UpSampling(3, 1)
    # print(ups(inp).shape)
