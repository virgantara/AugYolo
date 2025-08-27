import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- LKA (your code, lightly renamed) -----------------
class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn

class LKAAttn(nn.Module):
    """Your Attention(d_model) = 1x1 -> GELU -> LKA -> 1x1 + residual"""
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut


# ----------------- utilities (EfficientNet scaling) -----------------
def round_filters(filters, width_mult, divisor=8):
    filters *= width_mult
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, depth_mult):
    return int(math.ceil(repeats * depth_mult))  # official rule


# ----------------- Squeeze-and-Excitation -----------------
class SqueezeExcite(nn.Module):
    def __init__(self, in_ch, se_ratio=0.25):
        super().__init__()
        squeeze_ch = max(1, int(in_ch * se_ratio))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_ch, squeeze_ch, 1, bias=True)
        self.fc2 = nn.Conv2d(squeeze_ch, in_ch, 1, bias=True)

    def forward(self, x):
        s = self.avg(x)
        s = F.silu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


# ----------------- MBConv with optional LKA -----------------
class MBConv(nn.Module):
    """
    MBConv(in_ch -> mid_ch -> out_ch) with SE and optional LKA:
      - lka_where='mid'  : apply LKAAttn(mid_ch) after DW BN+SiLU, before SE
      - lka_where='out'  : apply LKAAttn(out_ch) after project BN
    """
    def __init__(self, in_ch, out_ch, expand_ratio, kernel, stride,
                 se_ratio=0.25, drop_connect=0.0,
                 use_lka=False, lka_where='mid'):
        super().__init__()
        assert lka_where in ('mid', 'out')
        self.has_residual = (stride == 1 and in_ch == out_ch)
        self.drop_connect = drop_connect
        self.use_lka = use_lka
        self.lka_where = lka_where

        mid_ch = in_ch * expand_ratio
        self._mid_ch = mid_ch
        self._out_ch = out_ch

        self.expand = nn.Sequential()
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch, eps=1e-3, momentum=0.01),
                nn.SiLU(inplace=True),
            )
        self.dw = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel, stride=stride, padding=kernel//2,
                      groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch, eps=1e-3, momentum=0.01),
            nn.SiLU(inplace=True),
        )
        self.se = SqueezeExcite(mid_ch, se_ratio=se_ratio)
        self.project = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
        )

        # (LKA modules are attached lazily to keep pretrain loading simple)
        self.lka_mid = None
        self.lka_out = None

    # allow adding LKA after loading the pretrained backbone
    def attach_lka(self, where='mid'):
        assert where in ('mid', 'out')
        if where == 'mid' and self.lka_mid is None:
            self.lka_mid = LKAAttn(self._mid_ch)
        if where == 'out' and self.lka_out is None:
            self.lka_out = LKAAttn(self._out_ch)

    def forward(self, x):
        y = self.expand(x)
        y = self.dw(y)
        if self.use_lka and self.lka_where == 'mid' and self.lka_mid is not None:
            y = self.lka_mid(y)
        y = self.se(y)
        y = self.project(y)
        if self.use_lka and self.lka_where == 'out' and self.lka_out is not None:
            y = self.lka_out(y)
        if self.has_residual:
            y = y + x
        return y


# ----------------- EfficientNet-B4 skeleton -----------------
class EfficientNetB4_LKA(nn.Module):
    """
    EfficientNet-B4 (width=1.4, depth=1.8) with optional LKA injection.
    lka_stages: list of stage indices (0..6) to augment, or None for all.
    lka_where : 'mid' (default) or 'out'.
    """
    def __init__(self, num_classes=1000, width_mult=1.4, depth_mult=1.8,
                 dropout=0.4, drop_connect=0.2,
                 use_lka=False, lka_stages=None, lka_where='mid'):
        super().__init__()
        self.use_lka = use_lka
        self.lka_stages = set(lka_stages) if lka_stages is not None else None
        self.lka_where = lka_where

        # base (B0) cfg: (exp, c_out, repeats, k, s)
        cfgs = [
            (1, 16,  1, 3, 1),
            (6, 24,  2, 3, 2),
            (6, 40,  2, 5, 2),
            (6, 80,  3, 3, 2),
            (6, 112, 3, 5, 1),
            (6, 192, 4, 5, 2),
            (6, 320, 1, 3, 1),
        ]
        se_ratio = 0.25

        in_ch = round_filters(32, width_mult)   # stem 32 -> 48
        self.stem = nn.Sequential(
            nn.Conv2d(3, in_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_ch, eps=1e-3, momentum=0.01),
            nn.SiLU(inplace=True)
        )

        blocks = []
        stage_idx = -1
        total_blocks = sum(round_repeats(r, depth_mult) for *_, r, _, _ in cfgs)
        b_idx = 0
        for (exp, c, r, k, s) in cfgs:
            stage_idx += 1
            out_ch = round_filters(c, width_mult)
            reps = round_repeats(r, depth_mult)
            for i in range(reps):
                stride = s if i == 0 else 1
                dropc = drop_connect * b_idx / max(1, total_blocks - 1)
                add_lka_here = self.use_lka and (self.lka_stages is None or stage_idx in self.lka_stages)
                blk = MBConv(in_ch, out_ch, expand_ratio=exp, kernel=k, stride=stride,
                             se_ratio=se_ratio, drop_connect=dropc,
                             use_lka=add_lka_here, lka_where=self.lka_where)
                blocks.append(blk)
                in_ch = out_ch
                b_idx += 1

        self.blocks = nn.Sequential(*blocks)
        head_ch = round_filters(1280, width_mult)  # 1280 -> 1792
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_ch, 1, bias=False),
            nn.BatchNorm2d(head_ch, eps=1e-3, momentum=0.01),
            nn.SiLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(head_ch, num_classes)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

    # attach LKA **after** loading pretrained weights (so loader alignment is safe)
    def enable_lka_after_load(self):
        if not self.use_lka:
            return
        stage = -1
        last_out = None
        for m in self.blocks:
            # stage changes whenever out_ch changes
            out_ch = m._out_ch
            if out_ch != last_out:
                stage += 1
                last_out = out_ch
            if self.lka_stages is None or stage in self.lka_stages:
                m.attach_lka(self.lka_where)

def _convs_and_bns_in_order(root: nn.Module):
    """Return a flat list of (module, idx) for all Conv2d/BN modules in preorder."""
    mods = []
    for i, m in enumerate(root.modules()):
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
            mods.append((m, i))
    return mods

def _next_bn_after(mods, start_idx, expected_channels):
    """Find the first BN after start_idx with matching num_features."""
    for m, i in mods:
        if i > start_idx and isinstance(m, nn.BatchNorm2d) and m.num_features == expected_channels:
            return m
    return None

def _split_mbconv_tv(s: nn.Module):
    """
    Heuristically parse a torchvision MBConv 's' into parts:
      expand_conv/bn, depthwise_conv/bn, se_reduce/se_expand, project_conv/bn
    Works across torchvision variants (no reliance on attribute names).
    """
    parts = dict(expand_conv=None, expand_bn=None,
                 depthwise_conv=None, depthwise_bn=None,
                 se_reduce=None, se_expand=None,
                 project_conv=None, project_bn=None)

    # Get all convs & BNs with their preorder indices
    linear_mods = _convs_and_bns_in_order(s)

    # Gather convs only with indices to maintain order
    convs = [(m, i) for m, i in linear_mods if isinstance(m, nn.Conv2d)]

    # Identify candidates by shape
    # expand: 1x1, groups=1, in!=out (if present)
    expand_cands = [(m, i) for m, i in convs if m.kernel_size == (1, 1) and m.groups == 1 and m.in_channels != m.out_channels]
    # depthwise: groups == in_channels == out_channels, kernel > 1
    dw_cands = [(m, i) for m, i in convs if m.groups == m.in_channels == m.out_channels and m.kernel_size != (1, 1)]
    # project: 1x1, groups=1, (usually after SE), last 1x1 conv
    proj_cands = [(m, i) for m, i in convs if m.kernel_size == (1, 1) and m.groups == 1]

    # pick by order: expand (first), depthwise (first after expand), project (last)
    if expand_cands:
        parts['expand_conv'], idx_expand = expand_cands[0]
        parts['expand_bn'] = _next_bn_after(linear_mods, idx_expand, parts['expand_conv'].out_channels)
    else:
        idx_expand = -1

    if dw_cands:
        # choose the first depthwise conv *after* expand (if expand exists)
        if idx_expand >= 0:
            dw_after = [(m, i) for m, i in dw_cands if i > idx_expand]
            parts['depthwise_conv'], idx_dw = (dw_after[0] if dw_after else dw_cands[0])
        else:
            parts['depthwise_conv'], idx_dw = dw_cands[0]
        parts['depthwise_bn'] = _next_bn_after(linear_mods, idx_dw, parts['depthwise_conv'].out_channels)
    else:
        idx_dw = -1

    if proj_cands:
        # choose the last 1x1 conv as project (it comes after SE)
        parts['project_conv'], idx_proj = proj_cands[-1]
        parts['project_bn'] = _next_bn_after(linear_mods, idx_proj, parts['project_conv'].out_channels)
    else:
        idx_proj = -1

    # SE convs: two 1x1 convs between DW and Project
    se_pool_seen = False
    se_convs_between = []
    for m in s.modules():
        if isinstance(m, nn.AdaptiveAvgPool2d):
            se_pool_seen = True
        if isinstance(m, nn.Conv2d) and se_pool_seen:
            se_convs_between.append(m)
    # fallback: if we didn't detect via pool, take 1x1 convs between dw and proj indices
    if not se_convs_between and idx_dw >= 0 and idx_proj >= 0:
        se_convs_between = [m for m, i in convs if i > idx_dw and i < idx_proj and m.kernel_size == (1,1)]

    if len(se_convs_between) >= 2:
        parts['se_reduce'], parts['se_expand'] = se_convs_between[0], se_convs_between[1]
    elif len(se_convs_between) == 1:
        parts['se_reduce'] = se_convs_between[0]

    return parts

@torch.no_grad()
def load_pretrained_from_torchvision_b4(custom_model, include_head: bool = False, verbose: bool = True):
    """
    Robust loader: copies weights from torchvision EfficientNet-B4 into your EfficientNetB4_LKA,
    without relying on attribute names in the TV MBConv.
    """
    from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
    tv = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT).eval()

    # --- stem ---
    custom_model.stem[0].weight.copy_(tv.features[0][0].weight)
    bn_src = tv.features[0][1]; bn_dst = custom_model.stem[1]
    bn_dst.weight.copy_(bn_src.weight); bn_dst.bias.copy_(bn_src.bias)
    bn_dst.running_mean.copy_(bn_src.running_mean); bn_dst.running_var.copy_(bn_src.running_var)

    # --- blocks (MBConv) ---
    tv_mbconvs = [m for m in tv.features.modules() if m.__class__.__name__.lower().startswith('mbconv')]
    my_mbconvs = [m for m in custom_model.blocks if isinstance(m, MBConv)]

    copied = 0
    for s, d in zip(tv_mbconvs, my_mbconvs):
        parts = _split_mbconv_tv(s)

        # expand
        if len(d.expand) and parts['expand_conv'] is not None:
            d.expand[0].weight.copy_(parts['expand_conv'].weight)
            if parts['expand_bn'] is not None:
                d.expand[1].weight.copy_(parts['expand_bn'].weight)
                d.expand[1].bias.copy_(parts['expand_bn'].bias)
                d.expand[1].running_mean.copy_(parts['expand_bn'].running_mean)
                d.expand[1].running_var.copy_(parts['expand_bn'].running_var)
            copied += 2

        # depthwise
        if parts['depthwise_conv'] is not None:
            d.dw[0].weight.copy_(parts['depthwise_conv'].weight)
        if parts['depthwise_bn'] is not None:
            d.dw[1].weight.copy_(parts['depthwise_bn'].weight)
            d.dw[1].bias.copy_(parts['depthwise_bn'].bias)
            d.dw[1].running_mean.copy_(parts['depthwise_bn'].running_mean)
            d.dw[1].running_var.copy_(parts['depthwise_bn'].running_var)
        copied += 2

        # SE
        if parts['se_reduce'] is not None:
            d.se.fc1.weight.copy_(parts['se_reduce'].weight)
            if parts['se_reduce'].bias is not None and d.se.fc1.bias is not None:
                d.se.fc1.bias.copy_(parts['se_reduce'].bias)
        if parts['se_expand'] is not None:
            d.se.fc2.weight.copy_(parts['se_expand'].weight)
            if parts['se_expand'].bias is not None and d.se.fc2.bias is not None:
                d.se.fc2.bias.copy_(parts['se_expand'].bias)
        copied += 2

        # project
        if parts['project_conv'] is not None:
            d.project[0].weight.copy_(parts['project_conv'].weight)
        if parts['project_bn'] is not None:
            d.project[1].weight.copy_(parts['project_bn'].weight)
            d.project[1].bias.copy_(parts['project_bn'].bias)
            d.project[1].running_mean.copy_(parts['project_bn'].running_mean)
            d.project[1].running_var.copy_(parts['project_bn'].running_var)
        copied += 2

    # --- head ---
    custom_model.head[0].weight.copy_(tv.features[-1][0].weight)
    bn_src = tv.features[-1][1]; bn_dst = custom_model.head[1]
    bn_dst.weight.copy_(bn_src.weight); bn_dst.bias.copy_(bn_src.bias)
    bn_dst.running_mean.copy_(bn_src.running_mean); bn_dst.running_var.copy_(bn_src.running_var)
    copied += 2

    # classifier (optional)
    if include_head and custom_model.fc.out_features == tv.classifier[1].out_features:
        custom_model.fc.weight.copy_(tv.classifier[1].weight)
        custom_model.fc.bias.copy_(tv.classifier[1].bias)
        copied += 1

    if verbose:
        print(f"[tv->custom B4] copied parts: {copied} (agnostic to TV attribute names)")

    # attach LKA after weights are in
    custom_model.enable_lka_after_load()


# ----------------- convenience factory -----------------
def build_efficientnet_b4_with_lka(num_classes=1000,
                                   use_pretrained=True,
                                   lka=True, lka_where='mid',
                                   lka_stages=(3, 4, 5, 6),  # later stages by default
                                   dropout=0.4):
    model = EfficientNetB4_LKA(
        num_classes=num_classes, width_mult=1.4, depth_mult=1.8,
        dropout=dropout, drop_connect=0.2,
        use_lka=lka, lka_stages=list(lka_stages) if lka else None, lka_where=lka_where
    )
    if use_pretrained:
        load_pretrained_from_torchvision_b4(model, include_head=(num_classes == 1000))
    else:
        model.enable_lka_after_load()
    return model


# ----------------- quick test -----------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = build_efficientnet_b4_with_lka(num_classes=3,
                                       use_pretrained=True,
                                       lka=True, lka_where='mid',
                                       lka_stages=(3,4,5,6)).to(device)
    x = torch.randn(1, 3, 380, 380, device=device)
    y = m(x)
    print("out:", y.shape)
    print("params(M):", sum(p.numel() for p in m.parameters())/1e6)
