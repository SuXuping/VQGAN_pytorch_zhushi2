import torch
import argparse
from model.vqgan_transformer import vqgan_transformer
from model.vqgan import VQGAN
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image


def main(args):
    train_logger = SummaryWriter(args.logger)
    ###构建数据
    from dataset.get_data import my_dataset
    train_datasets = my_dataset(args)
    batch_size = args.batch_size
    train_dataloader = DataLoader(train_datasets,batch_size=batch_size,shuffle=True,num_workers=4)
    ###构造训练参数
    lr = args.learning_rate
    model = vqgan_transformer(args).to(args.device)
    total_params = [param for name, param in model.gpt.named_parameters() if param.requires_grad == True]
    opt_transformer = AdamW(total_params,lr=lr,betas=(args.beta1,args.beta2),weight_decay=1e-04)
    for epoch in range(args.epochs):
        with tqdm(range(len(train_dataloader)),desc="training vqgantranformer") as pbar:
            for step_index,data in zip(pbar,train_dataloader):
                total_step_index = epoch * len(train_dataloader) + step_index
                data = data.to(args.device)
                logits,real_indices = model(data)
                loss = F.cross_entropy(logits.reshape(-1,args.num_codebook_vectors),real_indices)
                opt_transformer.zero_grad()
                loss.backward()
                opt_transformer.step()
                pbar.set_postfix(loss = np.round(loss.cpu().detach().numpy(),3))
                train_logger.add_scalar("tranformer_loss",loss,total_step_index)
        torch.save(model.state_dict(),os.path.join(args.checkpoint_path,f"vqgan_tranformer_{epoch}.pt"))
        with torch.no_grad():
            sample_batch_size = 4
            start_indices = torch.zeros((sample_batch_size, 0)).long().to("cuda")
            sos_tokens = torch.ones(sample_batch_size, 1) * args.sos_token
            sos_tokens = sos_tokens.long().to("cuda")
            predict_indices = model.sample(start_indix=start_indices,sos=sos_tokens,steps=256)  ### 256 = h * w,是vqgan训练时候经过encode之后得到特征图的宽高
            predict_imgs = model.z_to_img(predict_indices)  ### predict_indices为[b,256]
            fake_imgs = predict_imgs.add(1).mul(0.5)[:4]
            save_image(fake_imgs,os.path.join("results_transformer",f"{epoch}.jpg"),nrow=4)


if __name__ == "__main__":
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

    main(args)