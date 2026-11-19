"""CXS-YOLO modules retained for the public release."""
import math
import torch
import torch.nn as nn

from .block import C2f, C3, C3k, DFL
from .conv import Conv
from .head import Detect
from ultralytics.utils.tal import dist2bbox, make_anchors


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__(); self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x): return self.relu(x + 3.0) / 6.0


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__(); self.sigmoid = h_sigmoid(inplace)
    def forward(self, x): return x * self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__(); hidden = max(1, in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1); self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, hidden, 1, bias=False); self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden, in_planes, 1, bias=False); self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu1(self.fc1(self.avg_pool(x)))) + self.fc2(self.relu1(self.fc1(self.max_pool(x)))))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__(); p = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=p, bias=False); self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.conv1(torch.cat((x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]), 1)))


def _fixed_kernel(kind):
    if kind == "mean": return torch.ones(1, 1, 3, 3) / 9
    k = torch.full((1, 1, 3, 3), -1 / 8); k[..., 1, 1] = 0; return k


class MAC(nn.Module):
    def __init__(self, inplanes, outplanes, one=3, two=3, three=3, scales=4):
        super().__init__(); self.scales = scales; self.spx = outplanes // scales
        if outplanes % scales: raise ValueError("outplanes must be divisible by scales")
        self.inconv = Conv(inplanes, outplanes, act=False)
        self.conv1 = Conv(self.spx, self.spx, k=one, g=self.spx, act=False)
        self.conv2 = Conv(self.spx, self.spx, k=two, g=self.spx, act=False)
        self.conv3 = nn.Conv2d(self.spx, self.spx, 3, padding=1, groups=self.spx, bias=True)
        self.conv4 = nn.Conv2d(self.spx, self.spx, 3, padding=2, dilation=2, groups=self.spx, bias=True)
        self.conv5 = nn.BatchNorm2d(self.spx)
        self.ca = ChannelAttention(outplanes)
        self.sa = SpatialAttention()
        self.spatial_projection = nn.Conv2d(1, 1, kernel_size=1)
        nn.init.ones_(self.spatial_projection.weight)
        nn.init.zeros_(self.spatial_projection.bias)
        self.relu = nn.ReLU(inplace=True)
        with torch.no_grad():
            self.conv3.weight.copy_(_fixed_kernel("mean").repeat(self.spx, 1, 1, 1)); self.conv4.weight.copy_(_fixed_kernel("ring").repeat(self.spx, 1, 1, 1))
            self.conv3.bias.zero_(); self.conv4.bias.zero_()
        for p in (self.conv3.weight, self.conv4.weight): p.requires_grad_(False)
    def forward(self, x):
        projected_input = self.inconv(x)
        branches = torch.chunk(projected_input, self.scales, dim=1)
        x1, x2, x3, x4 = branches[:4]
        y2 = self.relu(self.conv1(x2))
        y3 = self.relu(self.conv2(x3 + y2))
        branch4_input = x4 + y3
        y4 = self.relu(self.conv5(self.conv3(branch4_input) + self.conv4(branch4_input)))
        features = torch.cat((x1, y2, y3, y4), dim=1)

        if not hasattr(self, "spatial_projection"):
            return self.relu(self.outconv(features) + projected_input)

        channel_refined = features * self.ca(features)
        spatial_weights = self.spatial_projection(self.sa(channel_refined))
        return channel_refined * spatial_weights


class C3k_MAC(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k); c_ = int(c2 * e); self.m = nn.Sequential(*(MAC(c_, c_) for _ in range(n)))


class C3k2_MCC(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e); self.m = nn.ModuleList(C3k_MAC(self.c, self.c, 2, shortcut, g) if c3k else MAC(self.c, self.c) for _ in range(n))


class LECA(nn.Module):
    def __init__(self, inp, reduction=32, min_channels=8, eps=1e-6, residual=True):
        super().__init__(); mip = max(min_channels, inp // reduction); self.eps = eps; self.residual = residual
        self.conv1 = nn.Conv2d(inp, mip, 1, bias=False); self.norm = nn.GroupNorm(1, mip, affine=False); self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, inp, 1, bias=False); self.conv_w = nn.Conv2d(mip, inp, 1, bias=False)
    def forward(self, x):
        identity=x; mean=x.mean((2,3),keepdim=True); x=(x-mean)/torch.sqrt((x-mean).pow(2).mean((2,3),keepdim=True)+self.eps)
        x_h=x.mean(3,keepdim=True); x_w=x.mean(2,keepdim=True).permute(0,1,3,2); y=self.act(self.norm(self.conv1(torch.cat((x_h,x_w),2)))); h,w=x_h.shape[2],x_w.shape[2]
        a_h=torch.sigmoid(self.conv_h(y[:,:,:h])); a_w=torch.sigmoid(self.conv_w(y[:,:,h:]).permute(0,1,3,2)); contrast=(identity-identity.mean((2,3),keepdim=True)).abs().mean(1,keepdim=True); prior=torch.sigmoid(contrast/(contrast.mean((2,3),keepdim=True)+self.eps)); att=a_h*a_w*prior
        return identity*(1+att) if self.residual else identity*att


class CAWD(nn.Module):
    def __init__(self, ch, group=16):
        super().__init__(); self.softmax=nn.Softmax(-1); self.attention=nn.Sequential(nn.AvgPool2d(3,1,1),Conv(ch,ch,1)); self.ds_conv=Conv(ch,ch*4,3,2,g=max(1,ch//group))
    def forward(self,x):
        b,c,h,w=x.shape; att=self.attention(x).reshape(b,c,h//2,2,w//2,2).permute(0,1,2,4,3,5).reshape(b,c,h//2,w//2,4); y=self.ds_conv(x).reshape(b,4,c,h//2,w//2).permute(0,2,3,4,1); return (y*self.softmax(att)).sum(-1)


class Scale(nn.Module):
    def __init__(self, scale=1.0): super().__init__(); self.scale=nn.Parameter(torch.tensor(scale,dtype=torch.float))
    def forward(self,x): return x*self.scale

class Conv_GN(nn.Module):
    def __init__(self,in_c,out_c,k=1,s=1,p=None,g=1,act=True):
        super().__init__(); self.conv=nn.Conv2d(in_c,out_c,k,s,k//2 if p is None else p,groups=g,bias=False); self.gn=nn.GroupNorm(g if g>1 else out_c,out_c); self.act=nn.SiLU() if act is True else (act if isinstance(act,nn.Module) else nn.Identity())
    def forward(self,x): return self.act(self.gn(self.conv(x)))

class DynamicScaleGenerator(nn.Module):
    def __init__(self,in_c,reduction=4):
        super().__init__(); h=max(1,in_c//reduction); self.gap=nn.AdaptiveAvgPool2d(1); self.conv1=nn.Conv2d(in_c,h,1,bias=False); self.act=nn.SiLU(); self.conv2=nn.Conv2d(h,in_c,1,bias=False); self.sigmoid=nn.Sigmoid()
    def forward(self,x): return self.sigmoid(self.conv2(self.act(self.conv1(self.gap(x)))))

class Detect_SALSCD(Detect):
    def __init__(self,nc=80,hidc=256,ch=()):
        super().__init__(nc,ch); self.reg_max=8; self.no=nc+4*self.reg_max; self.conv=nn.ModuleList(nn.Sequential(Conv_GN(x,hidc,3)) for x in ch); self.share_conv=nn.Sequential(Conv_GN(hidc,hidc,3,g=hidc),Conv_GN(hidc,hidc,1)); self.cv2=nn.Conv2d(hidc,4*self.reg_max,1); self.cv3=nn.Conv2d(hidc,nc,1); self.scale=nn.ModuleList(Scale(1.) for _ in ch); self.dfl=DFL(self.reg_max); self.dynamic_generators=nn.ModuleList(DynamicScaleGenerator(hidc) for _ in ch)
    def forward(self,x):
        for i in range(self.nl):
            f=self.conv[i](x[i]); f=self.share_conv(f)*self.dynamic_generators[i](f); x[i]=torch.cat((self.scale[i](self.cv2(f)),self.cv3(f)),1)
        if self.training:return x
        shape=x[0].shape; cat=torch.cat([z.view(shape[0],self.no,-1) for z in x],2); self.anchors,self.strides=(z.transpose(0,1) for z in make_anchors(x,self.stride,0.5)); box,cls=cat.split((self.reg_max*4,self.nc),1); return (torch.cat((dist2bbox(self.dfl(box),self.anchors.unsqueeze(0),xywh=True,dim=1)*self.strides,cls.sigmoid()),1),x)
    def bias_init(self):
        nn.init.constant_(self.cv2.bias, 1.0)
        nn.init.constant_(self.cv3.bias, math.log(5 / max(self.nc, 1) / (640 / 16) ** 2))
