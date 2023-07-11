import torch
import argparse
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from torchvision.utils import save_image
from dataset.get_data import my_dataset
from model.vqgan import VQGAN
from model.discriminator import Discriminator
from model.perceptual import Perceptual
from model.utils import weights_init
from torch.utils.tensorboard import SummaryWriter

def main(args):
    train_logger_path = os.path.join(args.logger_path,"train")
    train_logger = SummaryWriter(train_logger_path)
    ##产生数据集
    train_dataset = my_dataset(args)
    train_data_loader = DataLoader(train_dataset,batch_size=args.batch_size,pin_memory=True,shuffle=True)
    ##配置训练参数
    lr = args.learning_rate
    vq_model = VQGAN(args).to(args.device)
    dis_model = Discriminator(args).to(args.device)
    dis_model.apply(weights_init)
    perceptual_model = Perceptual().eval().to(args.device)
    opt_vq = torch.optim.Adam(vq_model.parameters(),lr=lr,betas=(args.beta1,args.beta2),eps=1e-08)
    opt_gan = torch.optim.Adam(dis_model.parameters(),lr=lr,betas=(args.beta1,args.beta2),eps=1e-08)    

    # opt_vq = torch.optim.Adam(
    #     list(vq_model.encoder.parameters()) +
    #     list(vq_model.decoder.parameters()) +
    #     list(vq_model.codebook.parameters()) +
    #     list(vq_model.quant_conv.parameters()) +
    #     list(vq_model.post_quant_conv.parameters()),
    #     lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
    # )
    # opt_gan = torch.optim.Adam(dis_model.parameters(),lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))
    
    for epoch in range(args.epochs):
        with tqdm(range(len(train_data_loader)),desc="training vagan") as pbar:
            for step_index, imgs in zip(pbar, train_data_loader):
                total_step_index = epoch * len(train_data_loader) + step_index
                imgs = imgs.to(args.device)
                # print(f"开始训练epoch={epoch}, step={step_index}/{len(train_data_loader)}")
                imgs_decode,_,q_loss = vq_model(imgs)  ###生成器
                imgs_fake = dis_model(imgs_decode) ### 判别器
                imgs_real = dis_model(imgs) ### 判别器
                disc_factor = vq_model.adopt_weight(args.disc_factor, epoch*len(train_data_loader) + step_index, args.disc_start)
                ###判别器的loss
                d_loss_real = torch.mean(F.relu(1.0 - imgs_real))
                d_loss_fake = torch.mean(F.relu(1.0 + imgs_fake))
                dis_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)
                g_loss = -torch.mean(imgs_fake)  ###对应论文公式5，用于判断生成的图片是否真实
                ###生成器的loss = 重建的L1loss + 感知损失 + q_loss，论文中公式4
                rec_loss_l1 = torch.abs(imgs - imgs_decode)
                perceptual_loss = perceptual_model(imgs,imgs_decode)
                perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss_l1
                perceptual_rec_loss = torch.mean(perceptual_rec_loss)
                generate_loss = perceptual_rec_loss + q_loss
                ### 参考论文中的公式6，判别器和生成器之间的权重
                λ = vq_model.calculate_lambda(perceptual_rec_loss,g_loss)
                vq_loss = generate_loss +  disc_factor * λ * g_loss
                train_logger.add_scalar("vq_loss",vq_loss,total_step_index)
                ### 更新生成器vqgan的参数
                opt_vq.zero_grad()
                vq_loss.backward(retain_graph=True)

                ### 更新判别器的参数
                opt_gan.zero_grad()
                dis_loss.backward()

                opt_gan.step()
                opt_vq.step()

                if step_index % 20 == 0:
                    with torch.no_grad():
                        real_fake_imgs = torch.cat((imgs[:4],imgs_decode.add(1).mul(0.5)[:4]))  ### imgs_decode范围[-1,1]，经过加1，乘0.5，变换到[0-1]之间，saveimage对【0-1】之间的tensor进行保存
                        # mean = (0.485, 0.456, 0.406)
                        # std = (0.229, 0.224, 0.225)
                        # for i in range(imgs_decode.size(0)):
                        #     img = imgs_decode[i].permute(1, 2, 0)
                        #     img = img.to("cpu") * torch.tensor(std) + torch.tensor(mean) 
                        #     imgs_decode[i] = img.permute(2,0,1)
                        # real_fake_imgs = torch.cat((imgs[:4],imgs_decode[:4]))  ### imgs_decode范围[-1,1]，经过加1，乘0.5，变换到[0-1]之间，saveimage对【0-1】之间的tensor进行保存    
                        save_image(real_fake_imgs,os.path.join("results",f"{epoch}_{step_index}.jpg"),nrow=4)
                pbar.set_postfix(
                        vq_loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                        dis_loss=np.round(dis_loss.cpu().detach().numpy().item(), 3),
                        perceptual_rec_loss=np.round(perceptual_rec_loss.cpu().detach().numpy().item(), 3),
                        q_loss=np.round(q_loss.cpu().detach().numpy().item(), 3),
                        g_loss1=np.round((disc_factor * λ * g_loss).cpu().detach().numpy().item(), 3),
                        g_loss=np.round(g_loss.cpu().detach().numpy().item(), 3),
                    )
                pbar.update(0)
        torch.save(vq_model.state_dict(),os.path.join("checkpoints",f"vqgan_epoch_{epoch}.pt"))
        torch.save(dis_model.state_dict(),os.path.join("checkpoints",f"dis_epoch_{epoch}.pt"))

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
    main(args)