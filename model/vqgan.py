import torch
import torch.nn as nn
import torch.nn.functional as F
from .codebook import Codebook
from .decoder import Decoder
from .encoder import encoder

class VQGAN(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        self.encoder = encoder(args)
        self.codebook = Codebook(args)
        self.decoder  = Decoder(args)
        self.quant_conv = nn.Conv2d(args.latent_dim,args.latent_dim,1)
        self.post_quant_conv = nn.Conv2d(args.latent_dim,args.latent_dim,1)

    # def forward(self,x):
    #     z = self.encoder(x)
    #     z = self.quant_conv(z)
    #     z_q,codebook_indices,q_loss = self.codebook(z)
    #     z_q = self.post_quant_conv(z_q)
    #     gen = self.decoder(z_q)
    #     return gen,codebook_indices,q_loss

    def forward(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        decoded_images = self.decoder(post_quant_conv_mapping)

        return decoded_images, codebook_indices, q_loss

    def encode(self,imgs):
        z = self.encoder(imgs)
        z = self.quant_conv(z)
        z_q,codebook_indices,q_loss = self.codebook(z)
        return z_q,codebook_indices,q_loss
    
    def decode(self,imgs):
        z_q = self.post_quant_conv(imgs)
        gen = self.decoder(z_q)
        return gen
    
    def calculate_lambda(self,perceptual_loss,gan_loss):
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight
        perceptual_loss_grad = torch.autograd.grad(perceptual_loss,last_layer_weight,retain_graph=True)[0]
        gan_loss_grad = torch.autograd.grad(gan_loss,last_layer_weight,retain_graph=True)[0]
        
        λ = torch.norm(perceptual_loss_grad) / (torch.norm(gan_loss_grad) + 1e-4)
        λ = torch.clamp(λ,0,1e4).detach()
        return 0.8 * λ
    
    @staticmethod
    def adopt_weight(disc_factor,step,start_step,value=0.):
        if step < start_step:
            # disc_factor = value
            disc_factor = disc_factor
        return disc_factor