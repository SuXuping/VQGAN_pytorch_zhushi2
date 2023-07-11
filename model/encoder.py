import torch
import torch.nn as nn
from .utils import ResidualBlock,NonLocalBlock,DownSampleBlock,GroupNorm,Swish


class encoder(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        channels = [128,128,128,256,256,512]
        num_res_blocks = 2
        resolution = 256
        atten_resolution = [16]
        layers = [nn.Conv2d(args.image_channels,channels[0],kernel_size=3,stride=1,padding=1)]
        for i in range(len(channels) - 1):
            in_channel = channels[i]
            out_channel = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channel,out_channel))
                in_channel = out_channel
                if resolution in atten_resolution:
                    layers.append(NonLocalBlock(in_channel))
            if i != len(channels)-2:
                layers.append(DownSampleBlock(channels[i+1]))
                resolution //= 2
        layers.append(ResidualBlock(channels[-1],channels[-1]))
        layers.append(NonLocalBlock(channels[-1]))
        layers.append(ResidualBlock(channels[-1],channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[-1],args.latent_dim,3,1,1))
        self.model = nn.Sequential(*layers)

    def forward(self,x):
        return self.model(x)
    
class init_agrs():
    def __init__(self) -> None:
        self.latent_dim = 128
        self.image_channel = 3


if __name__ == "__main__":
    input = torch.randint(0,255,(1,3,256,256),dtype=torch.float32)
    # input = torch.randn((1,3,256,256))
    args = init_agrs()
    net = encoder(args)
    result = net(input)
    print(result.shape)