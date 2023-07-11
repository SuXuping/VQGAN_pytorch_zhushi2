import os
import argparse
import torch
from torchvision import utils as vutils
from model.vqgan_transformer import vqgan_transformer
from tqdm import tqdm


parser = argparse.ArgumentParser(description="VQGAN")
parser.add_argument('--latent_dim', type=int, default=256, help='Latent dimension n_z.')
parser.add_argument('--image_size', type=int, default=256, help='Image height and width.)')
parser.add_argument('--num_codebook_vectors', type=int, default=1024, help='Number of codebook vectors.')
parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
parser.add_argument('--image_channels', type=int, default=3, help='Number of channels of images.')
parser.add_argument('--dataset_path', type=str, default='./data', help='Path to data.')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/', help='Path to checkpoint.')
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

###训练transformer的参数
parser.add_argument('--vqgan_model_path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
parser.add_argument('--pkeep', type=float, default=0.5, help='Percentage for how much latent codes to keep.')
parser.add_argument('--sos_token', type=int, default=0, help='Start of Sentence token.')
parser.add_argument('--vocab_size', type=int, default=1024, help='Number of codebook vectors.')
parser.add_argument('--block_size', type=int, default=512, help='Number of codebook vectors.')
parser.add_argument('--n_embed', type=int, default=256, help='Number of codebook vectors.')
parser.add_argument('--n_layer', type=int, default=12, help='Number of codebook vectors.')
parser.add_argument('--logger', type=str, default='./logger/train/transformer', help='Path to data.')
parser.add_argument('--n_unmasked', type=int, default=0, help='Path to data.')
args = parser.parse_args()
args.dataset_path = r"./roses"
args.vqgan_model_path = r"./checkpoints/vqgan_epoch_99.pt"

n = 100
model = vqgan_transformer(args).to("cuda")
# model.load_state_dict(torch.load(os.path.join("./checkpoints", "vqgan_tranformer_99.pt")))
model = model.load_state_dict(torch.load("/home/aigc/sxp/sxp_VQGAN/checkpoints/vqgan_tranformer_50.pt"))
print("Loaded state dict of Transformer")

for i in tqdm(range(n)):
    start_indices = torch.zeros((4, 0)).long().to("cuda")
    sos_tokens = torch.ones(start_indices.shape[0], 1) * 0
    sos_tokens = sos_tokens.long().to("cuda")
    sample_indices = model.sample(start_indices, sos_tokens, steps=256)
    sampled_imgs = model.z_to_img(sample_indices)
    print(f'第{i}次')
    vutils.save_image(sampled_imgs, os.path.join("results_transformer_0", f"transformer_{i}.jpg"), nrow=4)
