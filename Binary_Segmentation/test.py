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
    image = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(image)
    image = image.unsqueeze(0)
    image = image.to(device, dtype=torch.float32)

    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output)
    
    mask_image = prediction.cpu().squeeze().numpy()
    
    image = cv2.imread(image_path+img)
    h, w = image.shape[:2]
    mask_image = cv2.resize(mask_image, (w, h))
  
    # prediction = prediction.permute(1, 2, 0).cpu().numpy()
    print(mask_image)
    mask_image *=255
    mask_image = cv2.normalize(mask_image, mask_image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # value of mask pixels may be 255

    for y in range(mask_image.shape[0]):
        for x in range(mask_image.shape[1]):
            if mask_image[y][x] < 0.5:
                print(mask_image[y][x])
                mask_image[y][x] = 0
            else:
                print(mask_image[y][x])

                mask_image[y][x] = 255

    contours = cv2.findContours(
        mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)


    cv2.imshow("image", image)
    cv2.imshow("mask", mask_image)

    cv2.waitKey()

cv2.destroyAllWindows()
