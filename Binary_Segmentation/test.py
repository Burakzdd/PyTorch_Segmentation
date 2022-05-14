from typing import Mapping
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import torch.utils.data.dataloader
import numpy as np
import cv2
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)

model = model.to(device)
model.load_state_dict(torch.load(
    "/home/burakzdd/my_workspace/PyTorch_Segmentation/Binary_Segmentation/weights/model.pth"))
model.eval()

image_path = "/home/burakzdd/Desktop/work/torch_segmentation/segmentation_full_body_mads_dataset_1192_img/test/"

for img in os.listdir(image_path):
    image = Image.open(image_path+img)
    image = image.resize((256,256), resample=Image.NEAREST)
    image = transforms.ToTensor()(image)
    # image = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(image)
    image = image.unsqueeze(0)
    image = image.to(device, dtype=torch.float32)

    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output)
    
    mask_image = prediction.cpu().squeeze().numpy()
    
    image = cv2.imread(image_path+img)
    h, w = image.shape[:2]
    mask_image = cv2.resize(mask_image, (w, h))
  
    cv2.imshow("image", image)
    cv2.imshow("mask", mask_image)

    cv2.waitKey()

cv2.destroyAllWindows()
