import os
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import torch
import torchvision.models as models
import torchvision.models.segmentation as segmentation
from datetime import date
import random
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import MulticlassJaccardIndex as IoULoss
import sys


transform = transforms.Compose([
    transforms.Resize((512,256)),
    transforms.ToTensor(),
])

class CustomDataset(Dataset):
    def __init__(self,root,img_dir, mask_dir, transform=None):
        self.root = root
        self.img_dir = os.path.join(root ,img_dir)
        self.mask_dir = os.path.join(root,mask_dir)
        
        self.transform = transform
        self.images = []
        self.masks = []
        # TODO delete it
        for file in os.listdir(img_dir):
            img_path = os.path.join(img_dir,file)
            mask_path = os.path.join(mask_dir,file.replace('jpg','png'))
            self.images.append(img_path)
            self.masks.append(mask_path)

    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = self.masks[index]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") 
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return {'image':image, 'mask':mask}

    def __len__(self):
        return len(self.images)


root = '.'
image_dir = sys.argv[1] #'/kaggle/input/mapillary/Mapillary-Vistas-1000-sidewalks/training/images'
mask_dir = sys.argv[2] #'/kaggle/working/output'

val_img = sys.argv[3] #"/kaggle/input/mapillary/Mapillary-Vistas-1000-sidewalks/validation/images"
val_mask = sys.argv[4] #"/kaggle/working/val"
valdata = CustomDataset(root,val_img,val_mask,transform)
valloader = DataLoader(valdata,batch_size=8,shuffle=True)

model_path = sys.argv[5] 

dataset = CustomDataset(root,image_dir, mask_dir, transform)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

resnet34 = models.resnet34(pretrained=True)
# Create a DeepLabv3+ model with ResNet-34 as the backbone
deeplab_model = segmentation.deeplabv3_resnet50(pretrained=False, num_classes=8)
deeplab_model.load_state_dict(torch.load(model_path)) #load model trained on cityscape


#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

num_classes = 8
#batch_size = 5
learning_rate = 0.0001
num_epochs = 30

criterion = IoULoss(num_classes = num_classes).to(device)#nn.CrossEntropyLoss()
optimizer = optim.Adam(deeplab_model.parameters(), lr=learning_rate)

deeplab_model = deeplab_model.to(device)


best_val_loss = float('inf')
# Training loop
for epoch in tqdm(range(num_epochs)):
    deeplab_model.train()
    for batch in data_loader:
        images = batch['image']
        masks = batch['mask']
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = deeplab_model(images)
        
        masks = masks.squeeze(1)
        masks = masks.long()
        loss = criterion(outputs['out'], masks)
        loss.requires_grad = True
        loss.backward()
        optimizer.step()
    
    deeplab_model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for val_batch in valloader:
            val_images = val_batch['image'].to(device)
            masks = val_batch['mask'].to(device)
            outputs = deeplab_model(val_images)
            masks = masks.squeeze(1)
            masks = masks.long()
            loss = criterion(outputs['out'], masks)
            total_loss += loss.item() * images.size(0)
        avg_val_loss = total_loss / len(valloader.dataset)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f'Saving the model at Epoch [{epoch + 1}/{num_epochs}]')
            torch.save(deeplab_model.state_dict(), 'deeplab_model_tuned.pth')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Val Loss: {avg_val_loss}')
