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

model =smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,
    classes=65
)

model = model.to(device)
model.load_state_dict(torch.load("./weights/model.pth"))
model.eval()

image_path = "/home/burakzdd/my_workspace/PyTorch_Segmentation/MultiClass_Segmentation/dataset/test/image/"
mask_path = "/home/burakzdd/my_workspace/PyTorch_Segmentation/MultiClass_Segmentation/dataset/test/mask/"
for img in os.listdir(image_path):
    image = Image.open(image_path+img)
    image = image.resize((256,256), resample=Image.NEAREST)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(image)
    
    with torch.no_grad():
        image = image.unsqueeze(0)
        image = image.to(device)#, dtype=torch.float32)
        
        output = model(image)
        prediction = torch.argmax(output,1)
    prediction = prediction.permute(1,2,0)
    
    mask_out = prediction.cpu().numpy()
    mask_out = mask_out.astype("uint8")
    image = cv2.imread(image_path+img)
    mask_image = cv2.imread(mask_path+img)
    
    w,h = image.shape[:2]
    mask_out = cv2.resize(mask_out, (h, w))
    cv2.imshow("img",image)
    cv2.imshow("mask",mask_image)
    cv2.imshow("output", mask_out)
    
    cv2.waitKey()
    
cv2.destroyAllWindows()
