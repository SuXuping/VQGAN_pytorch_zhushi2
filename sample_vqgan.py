import torch
from model.vqgan import VQGAN
import argparse
from torchvision.utils import save_image
import cv2
from dataset.get_data import my_dataset

def init_args():
    parser = argparse.ArgumentParser(description="VQGAN")
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent_dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image_size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num_codebook_vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image_channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset_path', type=str, default='./roses', help='Path to data (default: /data)')
    parser.add_argument('--logger_path', type=str, default='./logger', help='Path to logger (default: /logger)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch_size', type=int, default=6, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=2e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc_start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc_factor', type=float, default=1., help='')
    parser.add_argument('--rec_loss_factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual_loss_factor', type=float, default=1., help='Weighting factor for perceptual loss.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = init_args()
    net = my_dataset(args)
    image = net.__getitem__(160)

    noise = torch.from_numpy(image).unsqueeze(0)
    net = VQGAN(args)
    state_dict = torch.load("/home/aigc/sxp/sxp_VQGAN/checkpoints/vqgan_epoch_40.pt")
    net.load_state_dict(state_dict)
    pred = net(noise)[0]
    real_fake_imgs = torch.cat((noise[:4],pred.add(1).mul(0.5)[:4]))  ### imgs_decode范围[-1,1]，经过加1，乘0.5，变换到[0-1]之间，saveimage对【0-1】之间的tensor进行保存
    save_image(real_fake_imgs,"image.jpg")