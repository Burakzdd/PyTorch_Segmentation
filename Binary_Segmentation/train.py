import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm
from dataset import Custom_Dateset
import torch.utils.data.dataloader

img_path = "/home/burakzdd/Desktop/work/torch_segmentation/segmentation_full_body_mads_dataset_1192_img/images/"
mask_path = "/home/burakzdd/Desktop/work/torch_segmentation/segmentation_full_body_mads_dataset_1192_img/masks/"
batch_size = 4
epochs = 30

train_dataset = Custom_Dateset(img_path, mask_path, input_size = (256,256))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True,num_workers=8)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:"+str(device))

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

mean_losses = []

for epoch in range(epochs):
    running_loss = []
    loop = tqdm(enumerate(train_loader),total=len(train_loader))
    
    for batch_index, (image,mask) in loop:
        image, mask = image.to(device), mask.to(device)
        output = model(image)
        mask = mask.float()
        mask = mask.unsqueeze(0)
        mask = mask.permute(1, 0, 2, 3)
        loss = criterion(output,mask)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())
        mean_loss = sum(running_loss) / len(running_loss)
        loop.set_description(f'Epochs: [{epoch+1}/{epochs}]')
        loop.set_postfix(batch_loss = loss.item(), mean_loss=mean_loss, lr = optimizer.param_groups[0]["lr"])

    torch.save(model.state_dict(), "/home/burakzdd/my_workspace/PyTorch_Segmentation/Binary_Segmentation/weights/model.pth")