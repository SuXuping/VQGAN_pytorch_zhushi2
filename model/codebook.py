import torch
import torch.nn as nn

class Codebook(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = args.latent_dim
        self.beta = args.beta

        self.embedding = nn.Embedding(self.num_codebook_vectors,self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)
    
    def forward(self,z):  ### z:[1, latent_dim, 16, 16]
        z = z.permute(0,2,3,1).contiguous()
        z_flatten = z.view(-1,self.latent_dim)  ### z:[b*h*w, latent_dim]
        ###d:[b*h*w, num_codebook_vectors]
        d = torch.sum(z_flatten**2,dim=1,keepdim=True) + torch.sum(self.embedding.weight**2,dim=1) - 2*(torch.matmul(z_flatten,self.embedding.weight.t()))

        min_encoding_indices = torch.argmin(d,dim=1)### [b*h*w]
        z_q = self.embedding(min_encoding_indices) ### ([b*h*w],latent_dim)
        z_q = z_q.view(z.shape)  ###(b,h,w,latent_dim)

        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)
        
        z_q = z + (z_q - z).detach()  ###(z_q - z).detach()是没有梯度的。所以z_q的梯度完全由z决定，也就是把z的梯度直接复制给z_q

        z_q = z_q.permute(0,3,1,2).contiguous()  ###(b,latent_dim,h,w)

        return z_q,min_encoding_indices,loss
    
class init_args():
    def __init__(self) -> None:
        self.beta = 0.5
        self.latent_dim = 128
        self.num_codebook_vectors = 1000

if __name__ == "__main__":
    args = init_args()
    z = torch.randn((2,args.latent_dim,16,16))
    net = Codebook(args)
    z_q,min_encoding_indices,loss = net(z)
    print(z_q.shape)
