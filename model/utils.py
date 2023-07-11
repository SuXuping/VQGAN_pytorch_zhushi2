import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm(nn.Module):
    def __init__(self,channels) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=32,num_channels=channels,eps=1e-6,affine=True)
    def forward(self,x):
        return self.gn(x)

class Swish(nn.Module):
    def forward(self,x):
        return x*torch.sigmoid(x)

class ResidualBlock(nn.Module):
    def __init__(self,in_channel,out_channel) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block = nn.Sequential(
            GroupNorm(in_channel),
            Swish(),
            nn.Conv2d(in_channel,out_channel,3,1,1),
            GroupNorm(out_channel),
            Swish(),
            nn.Conv2d(out_channel,out_channel,3,1,1)
        )
        if in_channel != out_channel:
            self.channel_up = nn.Conv2d(in_channel,out_channel,1,1,0)

    def forward(self,x):
        if self.in_channel != self.out_channel:
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)
        
class NonLocalBlock(nn.Module):
    def __init__(self,channel) -> None:
        super().__init__()
        self.in_channel = channel
        self.gn = GroupNorm(channel)
        self.q = nn.Conv2d(channel,channel,1,1,0)
        self.k = nn.Conv2d(channel,channel,1,1,0)
        self.v = nn.Conv2d(channel,channel,1,1,0)

    def forward(self,x):
        h_ = self.gn(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b,c,h,w = q.shape

        q = q.reshape(b,c,h*w)
        k = k.reshape(b,c,h*w)
        v = v.reshape(b,c,h*w)

        q = q.permute(0,2,1) ###变换维度为了与K点乘

        attn = torch.bmm(q,k)  ##shape = [b,h*w,h*w]
        attn = attn * (int(c) ** (-0.5))
        attn = F.softmax(attn,dim=2)
        attn = attn.permute(0,2,1)

        A = torch.bmm(v,attn)
        A = A.reshape(b,c,h,w)

        return x + A

class DownSampleBlock(nn.Module):
    def __init__(self,channel) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channel,channel,3,2,0)
    def forward(self,x):
        pad = (0,1,0,1)
        x = F.pad(x,pad,mode="constant",value=0)
        return self.conv(x)
    
class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0)
        return self.conv(x)
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)