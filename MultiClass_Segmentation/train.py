import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import Custom_Dateset
import torch.utils.data.dataloader


img_path = "/home/burakzdd/my_workspace/PyTorch_Segmentation/MultiClass_Segmentation/dataset/IMAGES/"
mask_path = "/home/burakzdd/my_workspace/PyTorch_Segmentation/MultiClass_Segmentation/dataset/MASKS/"
batch_size = 4
epochs = 80

train_dataset = Custom_Dateset(img_path, mask_path, input_size=(256, 256))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

device = torch.device("cuda:0")
print("Device:"+str(device))

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes= 64
)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

mean_losses = []

for epoch in range(epochs):
    running_loss = []
    loop = tqdm(enumerate(train_loader), total=len(train_loader))

    for batch_index, (image, mask) in loop:
        image, mask = image.to(device), mask.to(device)
        output = model(image)

   
        loss = criterion(output, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())
        mean_loss = sum(running_loss) / len(running_loss)
        loop.set_description(f'Epochs: [{epoch+1}/{epochs}]')
        loop.set_postfix(batch_loss=loss.item(), mean_loss=mean_loss,
                         lr=optimizer.param_groups[0]["lr"])

    torch.save(model.state_dict(
    ), "/home/burakzdd/my_workspace/PyTorch_Segmentation/MultiClass_Segmentation/weights/model.pth")

