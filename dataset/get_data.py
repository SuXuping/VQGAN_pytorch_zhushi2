from typing import Any
from torch.utils.data import DataLoader,Dataset
import os
from PIL import Image
import numpy as np
import albumentations
from torchvision import transforms
from torchvision.transforms import Resize,CenterCrop,ToTensor,Normalize

# class my_dataset(Dataset):
#     def __init__(self,args,training=True) -> None:
#         super().__init__()
#         self.all_images = [os.path.join(args.dataset_path,name) for name in os.listdir(args.dataset_path)]
#         self.training = training
#         if self.training:
#             self.data_transform = transforms.Compose(
#                 [transforms.Resize(args.image_size),
#                 transforms.CenterCrop(args.image_size),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
#             )
#         else:
#             self.data_transform = transforms.Compose(
#                 [transforms.Resize(args.image_size),
#                 transforms.CenterCrop(args.image_size),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
#             )
            
#     def __len__(self):
#         return len(self.all_images)

#     def pre_process_image(self,image_path):
#         image = Image.open(image_path)
#         if image.mode != "RGB":
#             image = image.convert("RGB")
#         image = self.data_transform(image)
#         return image
    
#     def __getitem__(self, index) -> Any:
#         example = self.pre_process_image(self.all_images[index])
#         return example


class my_dataset(Dataset):
    def __init__(self, args, size=None):
        self.size = args.image_size

        # self.images = [os.path.join(path, file) for file in os.listdir(path)]
        self.images = [os.path.join(args.dataset_path,name) for name in os.listdir(args.dataset_path)]
        self._length = len(self.images)

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        return example

class init_args():
    def __init__(self) -> None:
        self.dataset_path = "/home/aigc/sxp/sxp_VQGAN/roses/"
        self.image_size = 256

if __name__ == "__main__":
    args = init_args()
    net = my_dataset(args)
    image = net.__getitem__(125)
    print(type(image))

    one_image_path = "/home/aigc/sxp/sxp_VQGAN/roses/12240303_80d87f77a3_n.jpg"
    one_image = Image.open(one_image_path)
    one_image1 = Resize(one_image)