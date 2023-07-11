import os
import sys
project_root = "/home/aigc/sxp/sxp_VQGAN/"
sys.path.append(project_root)
import torch
import torch.nn as nn
from model.vqgan import VQGAN
from model.gpt import GPT
import argparse
import torch.nn.functional as F
import math

class vqgan_transformer(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.sos_token = args.sos_token
        self.vqgan = self.load_vqgan(args)
        self.transformer_config = {
            "vocab_size":args.vocab_size,   ###字典的长度
            "block_size":args.block_size,   ###一个输入的句子的最多的token数
            "n_embed":args.n_embed,   ###每个字的维度
            "n_layer":args.n_layer,  ###多少层transformer encoder
            }
        self.gpt = GPT(**self.transformer_config)
        self.pkeep = args.pkeep
        self.sos_token = args.sos_token
        self.device = args.device

    def forward(self,imgs):
        ### 通过训练好的额vqgan得到图片对应的indices
        codebook_mapping, real_indices, q_loss = self.vqgan.encode(imgs)
        codebook_indices = real_indices.view(codebook_mapping.shape[0],-1)
        ### 生成sos_tokens
        sos_tokens = torch.ones(imgs.shape[0],1,dtype=torch.int64) * self.sos_token
        sos_tokens = sos_tokens.to(self.device)
        ### 随机替换codebook_indices生成new_indices
        mask = torch.bernoulli(self.pkeep * torch.ones(codebook_indices.shape)).to(self.device)   #### 按比例产生0或1的mask
        mask = mask.round().to(torch.int64)
        rand_indices = torch.randint_like(codebook_indices,self.transformer_config["vocab_size"])
        new_indices = mask * codebook_indices + (1 - mask) * rand_indices
        new_indices = torch.cat((sos_tokens, new_indices), dim=1)
        ### 将new_indices送入gpt中得到logits
        logits = self.gpt(new_indices[:,:-1])
        return logits,real_indices

    @staticmethod
    def load_vqgan(args):
        net = VQGAN(args)
        static_dict = torch.load(args.vqgan_model_path)
        net.load_state_dict(static_dict)
        for name,param in net.named_parameters():
            param.requires_grad = False
        return net.eval()
    
    @torch.no_grad()
    def sample(self,start_indix,sos,steps):
        self.gpt.eval()
        input = torch.cat((start_indix,sos),dim=-1)
        for step in range(steps):
            logits = self.gpt(input)
            logits = logits[:,-1,:]   ###只取返回的logits的最后一个token做softmax()，前面几个不管，保证每次只生成下一个token
            next_indix = torch.argmax(F.softmax(logits,dim=-1),dim=-1).reshape(input.shape[0],-1)
            # next_indix = torch.multinomial(F.softmax(logits,dim=-1),1)
            input = torch.cat((input,next_indix),dim=1)
        output = input[:,sos.shape[1]:]  ###最后得到的输出是在头部包含有sos的token的，所以这里需要把sos的token去除掉。
        self.gpt.train()
        return output
    @torch.no_grad()
    def z_to_img(self,z_indices):
        z_embedding = self.vqgan.codebook.embedding(z_indices)  ### ([b*h*w],latent_dim)
        b, h_w, laten_dim= z_embedding.shape
        h = int(math.sqrt(h_w))
        w = int(math.sqrt(h_w))  
        z_embedding = z_embedding.reshape(b,h,w,laten_dim).permute(0,3,1,2).contiguous()   ###(b,h,w,latent_dim) --> (b,latent_dim,h,w)
        decode_img = self.vqgan.decode(z_embedding)
        return decode_img
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent_dim', type=int, default=256, help='Latent dimension n_z.')
    parser.add_argument('--image_size', type=int, default=256, help='Image height and width.)')
    parser.add_argument('--num_codebook_vectors', type=int, default=1024, help='Number of codebook vectors.')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
    parser.add_argument('--image_channels', type=int, default=3, help='Number of channels of images.')
    parser.add_argument('--dataset_path', type=str, default='./data', help='Path to data.')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch_size', type=int, default=8, help='Input batch size for training.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', type=float, default=2.25e-05, help='Learning rate.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param.')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param.')
    parser.add_argument('--disc_start', type=int, default=10000, help='When to start the discriminator.')
    parser.add_argument('--disc_factor', type=float, default=1., help='Weighting factor for the Discriminator.')
    parser.add_argument('--l2_loss_factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual_loss_factor', type=float, default=1., help='Weighting factor for perceptual loss.')

    parser.add_argument('--pkeep', type=float, default=0.5, help='Percentage for how much latent codes to keep.')
    parser.add_argument('--sos_token', type=int, default=0, help='Start of Sentence token.')


    parser.add_argument('--vocab_size', type=int, default=1024, help='Number of codebook vectors.')
    parser.add_argument('--block_size', type=int, default=512, help='Number of codebook vectors.')
    parser.add_argument('--n_embed', type=int, default=256, help='Number of codebook vectors.')
    parser.add_argument('--n_layer', type=int, default=12, help='Number of codebook vectors.')
    args = parser.parse_args()
    args.dataset_path = r"./roses"
    args.vqgan_model_path = r"./checkpoints/vqgan_epoch_99.pt"

    net = vqgan_transformer(args).to("cuda")
    # input = torch.randint(0,255,(4,3,256,256),dtype=torch.float32)
    # # input = torch.randn((4,3,256,256))

    # logits,real_indices = net(input)
    # print(logits.shape)
    # print(real_indices.shape)

    net.gpt.to("cuda")
    sample_batch_size = 4
    start_indices = torch.zeros((sample_batch_size, 0)).long().to("cuda")
    sos_tokens = torch.ones(sample_batch_size, 1) * args.sos_token
    sos_tokens = sos_tokens.long().to("cuda")
    predict_indices = net.sample(start_indices,sos_tokens,steps=256)
    print(predict_indices.shape)
    new_imgs = net.z_to_img(predict_indices)
    print(new_imgs)