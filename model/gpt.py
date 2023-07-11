import torch
import torch.nn as nn
from torchvision.models import vgg16
import math
from torch.nn import functional as F


class init_config():
    def __init__(self,vocab_size,block_size,**kwargs):  ###kwargs传入的是一个不定长度的字典
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self,k,v)   ###setattr设定类变量和对应的值

class CausalSelfAttention(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        assert config.n_embed % config.n_head == 0, "token的维度必须整除于head的个数"
        self.head = config.n_head
        self.key = nn.Linear(config.n_embed,config.n_embed)
        self.query = nn.Linear(config.n_embed,config.n_embed)
        self.value = nn.Linear(config.n_embed,config.n_embed)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.rsid_drop = nn.Dropout(config.resid_pdrop)
        # 生成斜三角mask
        mask = torch.tril(torch.ones(config.block_size,config.block_size))
        self.register_buffer("mask",mask.view(1,1,config.block_size,config.block_size))
        if hasattr(config,"n_unmasked"):   ###根据定义的unmask部分设置为1
            mask[:config.n_unmasked,:config.n_unmasked] = 1
        # out projection
        self.proj = nn.Linear(config.n_embed,config.n_embed)

    def forward(self,x):
        B,T,C = x.shape
        q = self.query(x).view(B,T,self.head,int(C/self.head)).transpose(1,2)
        k = self.key(x).view(B,T,self.head,int(C/self.head)).transpose(1,2)
        v = self.value(x).view(B,T,self.head,int(C/self.head)).transpose(1,2)

        attn = (q @ k.transpose(-1,-2)) * (1.0 / math.sqrt(k.size(-1)))
        ### 将attn给mask掉
        attn = attn.masked_fill(self.mask[:,:,:T,:T] == 0,float('-inf'))  ####将mask为0的地方，填充为负无穷大，在后续计算softmask的时候，-inf的概率接近为0
        attn = F.softmax(attn,dim=-1)
        attn = self.attn_drop(attn)

        y = (attn @ v).transpose(1,2).contiguous().view(B,T,C)
        y = self.proj(y)
        y = self.rsid_drop(y)
        return y


class Block(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed),
            nn.Linear(4 * config.n_embed, config.n_embed),
        )

    def forward(self,x):
        x = self.ln1(x)
        attn_y = self.ln1(x + self.attn(x))  ### add + norm
        y = self.ln2(attn_y + self.mlp(attn_y))  ### feedforward + add + norm
        return y

class GPT(nn.Module):
    def __init__(self,vocab_size,block_size,n_layer=12,n_head=8,n_embed=256,
                 embed_pdrop=0.,resid_pdrop=0.,attn_pdrop=0.,n_unmasked=0) -> None:
        super().__init__()
        config = init_config(vocab_size=vocab_size,block_size=block_size,n_layer=n_layer,n_head=n_head,n_embed=n_embed,
                embed_pdrop=embed_pdrop,resid_pdrop=resid_pdrop,attn_pdrop=attn_pdrop,n_unmasked=n_unmasked)
        ###生成字典向量
        self.token_embed = nn.Embedding(config.vocab_size,config.n_embed)
        ###定义可学习的位置向量
        self.pos_embed = nn.Parameter(torch.zeros(1,config.block_size,config.n_embed))   ###(1 * 512 * 1024)
        self.drop = nn.Dropout(config.embed_pdrop)
        ### transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        ### decoder head
        self.ln = nn.LayerNorm(config.n_embed)
        self.head = nn.Linear(config.n_embed,config.vocab_size) ###对得到的特征进行映射到vocab_size对应的indice
        self.block_size = config.block_size
    def forward(self,idx):
        token_embedding = self.token_embed(idx)
        t = token_embedding.shape[1]
        assert t < self.block_size, "t=h*w,t的数值必须小于文本的长度即block_size"
        pos_embedding =  self.pos_embed[:,t,:]
        input_data = token_embedding + pos_embedding
        y = self.blocks(input_data)  ###经过多层transformer的encoder层抽取特征之后得到y
        y = self.ln(y)
        logits = self.head(y)  ### 得到抽取后的特征映射搭配vocab_size对应的索引indices,shape为【batch_size,(h*w),vocab_size】
        return logits

if __name__ == "__main__":
    indices = torch.randint(0,1024,(8,256)).to("cuda")
    net = GPT(vocab_size=1024,block_size=512,n_layer=12,n_head=8,n_embed=256,embed_pdrop=0.1,resid_pdrop=0.1,attn_pdrop=0.1,n_unmasked=0).to("cuda")
    logits = net(indices)
    print(logits.shape)