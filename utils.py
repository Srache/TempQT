import torch
import torch.nn as nn
import numpy as np
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import config


def rank_loss(p, y, num_images, eps=1e-6, is_norm=True, device=None):
    loss = torch.zeros(1, device=device)

    if num_images < 2:
        return loss
    dp = torch.abs(p)
    index = torch.arange(num_images)
    combinations = torch.combinations(index, 2)
    combinations_count = max(1, len(combinations))
    for i, j in combinations:
        rl = torch.clamp_min(-(y[i] - y[j]) * (p[i] - p[j]) / (torch.abs(y[i] - y[j]) + eps), min=0)
        loss += rl / max(dp[i], dp[j])  # normalize by maximum value
    if is_norm:
        loss = loss / combinations_count  # mean
    return loss


# ========================================================
# load pretrain model
# import os
# import model_pretrain
# load_path = config.CKPT_P.format('30')
# load_path = os.path.join(config.MODEL_PATH_P, load_path)
# ckpt = torch.load(load_path, map_location=config.DEVICE)
# model_err = model_pretrain.vit_IQAModel().to(config.DEVICE)
# model_err.load_state_dict(ckpt['state_dict'])
# model_err.requires_grad_(False)
# ========================================================
def calc_coefficient(dataloader_test, model, device):
    a = []
    b = []
    device = device

    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(dataloader_test)):
            batch_size = x.shape[0]
            y = y.reshape(batch_size, -1)
            x, y = x.to(device).float(), y.to(device).float()
            e, _ = model_err(x)
            inp = (x, e)
            p, _ = model(inp)
            a.append(y.cpu().float())
            b.append(p.cpu().float())
            # print(a)
            # print(b)
            #
            # import sys
            # sys.exit()
        a = np.vstack(a)
        b = np.vstack(b)
        a = a[:, 0]
        b = b[:, 0]

        a = np.reshape(a, (-1, config.TEST_PATCH_NUM))
        b = np.reshape(b, (-1, config.TEST_PATCH_NUM))
        a = np.mean(a, axis=1)
        b = np.mean(b, axis=1)
        
        sp = spearmanr(a, b)[0]
        pl = pearsonr(a, b)[0]
    model.train()
    return sp, pl


def calc_coefficient_ab(dataloader_test, model, device):
    a = []
    b = []
    device = device

    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(dataloader_test)):
            batch_size = x.shape[0]
            y = y.reshape(batch_size, -1)
            x, y = x.to(device).float(), y.to(device).float()
            p, _ = model(x)
            a.append(y.cpu().float())
            b.append(p.cpu().float())
        a = np.vstack(a)
        b = np.vstack(b)
        a = a[:, 0]
        b = b[:, 0]

        a = np.reshape(a, (-1, config.TEST_PATCH_NUM))
        b = np.reshape(b, (-1, config.TEST_PATCH_NUM))
        a = np.mean(a, axis=1)
        b = np.mean(b, axis=1)

        sp = spearmanr(a, b)[0]
        pl = pearsonr(a, b)[0]
    model.train()
    return sp, pl


def lr_scheduler(optimizer, epoch, lr_decay_epoch=8):
    '''
    Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs.
    '''
    decay_rate = 0.9 ** (epoch // lr_decay_epoch)

    # if epoch % lr_decay_epoch == 0:
        # print(f'decay_rate is set to {decay_rate}')

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

    return optimizer


def save_checkpoint(model, optimizer, filename="iqa_pretrain_kadid10k.pt"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        # self.p_conv.register_backward_hook(self._set_lr)
        self.p_conv.register_full_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset


class ChannelAttention(nn.Module):
    def __init__(self, in_channel=512, ratio=16):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channel // ratio, in_channel, 1, bias=False)
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc(self.max_pool(x))
        avg_out = self.fc(self.avg_pool(x))

        y = max_out + avg_out
        y = self.act(y)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super(SpatialAttention, self).__init__()
        assert k in (3, 7), 'Kernel size must be 3 or 7!'
        p = (k-1) // 2
        self.conv = nn.Conv2d(2, 1, k, padding=p, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.act(self.conv(y))
        return x * y


if __name__ == '__main__':
    x = torch.randn(5, 512, 100, 100)
    ca = ChannelAttention()
    sa = SpatialAttention()
    out_sa = sa(x)
    out_ca = ca(x)
    print(out_sa.shape)
    print(out_ca.shape)

    print('=' * 30)
    dconv = DeformConv2d(512, 3, 3)
    out = dconv(x)
    print(out.shape)


