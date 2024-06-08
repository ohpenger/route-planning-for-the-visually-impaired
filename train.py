import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.models as models
import torchvision.models.segmentation as segmentation
from torchmetrics.classification import MulticlassJaccardIndex as IoULoss
from tqdm.notebook import tqdm
from datetime import date
import sys
import numpy as np

labels = [0,7,8,19,20,24,26,33]
nolabels = [x for x in range(-1,34) if x not in labels]

transform = transforms.Compose([
    #transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

class CustomDataset(datasets.ImageFolder):
    def __init__(self, root,img_dir, mask_dir, transform=None):
        self.root = root
        self.img_dir = os.path.join(root,img_dir)
        self.mask_dir = os.path.join(root,mask_dir)
        self.city_dir = os.listdir(self.img_dir)
        
        self.transform = transform
        self.images = []
        self.masks = []
        for city in self.city_dir:
            city_img = os.path.join(self.img_dir,city)
            city_mask = os.path.join(self.mask_dir,city)
            
            for file in os.listdir(city_img):
                img_path = os.path.join(city_img,file)
                mask_path = os.path.join(city_mask,file)
                self.images.append(img_path)
                self.masks.append(mask_path)
                
    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = self.masks[index].replace('_leftImg8bit.png', '_gtFine_labelIds.png')
        image = Image.open(img_path).convert("RGB").resize((256,256))
        mask = Image.open(mask_path).convert("L").resize((256,256))
        mask = np.array(mask)
        for i in nolabels:
            mask[mask == i] = 0
        mask[mask > 33] = 0
        for i in range(1,len(labels)):
            mask[mask == labels[i]] = i 
        if self.transform is not None:
            image = self.transform(image)
            mask = torch.from_numpy(mask)

        return {'image':image, 'mask':mask}

    def __len__(self):
        return len(self.images)

root = ''
image_dir = sys.argv[1] #'/kaggle/input/images/leftImg8bit_trainvaltest/leftImg8bit/train' 
mask_dir = sys.argv[2] #'/kaggle/input/image-segmentation/gtFine_trainvaltest/gtFine/train'

val_img = sys.argv[3] #"/kaggle/input/images/leftImg8bit_trainvaltest/leftImg8bit/val"
val_mask = sys.argv[4] #"/kaggle/input/image-segmentation/gtFine_trainvaltest/gtFine/val"

dataset = CustomDataset(root,image_dir, mask_dir, transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
valdata = CustomDataset(root,val_img,val_mask,transform)
valloader = DataLoader(valdata,batch_size=32,shuffle=True)




# Create a DeepLabv3+ model with ResNet-50 as the backbone
deeplab_model = segmentation.deeplabv3_resnet50(pretrained=False, num_classes=8, weights_backbone =  models.ResNet50_Weights.DEFAULT)

classifier = list(deeplab_model.classifier.children())
classifier.append(nn.Softmax(dim=1))
deeplab_model.classifier = nn.Sequential(*classifier)

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

weights = np.array([0.54388354, 0.32192455, 0.05298849, 0.00495117, 0.00775358,
       0.00951798, 0.05644199, 0.00253869])
weights = [1 / (8 * i) for i in weights]
weights = np.array(weights)/sum(weights)
#weights = np.array(avg)/len(data_loader)
loss_weights = torch.from_numpy(weights).float().to(device)

num_classes = 8
#batch_size = 5
learning_rate = 1e-4
num_epochs = 50
weight_decay = 1e-5

criterion = nn.CrossEntropyLoss(weight=loss_weights)
optimizer = optim.Adam(deeplab_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
iou = IoULoss(num_classes=8).to(device)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#nn.DataParallel(deeplab_model)
deeplab_model = deeplab_model.to(device)


best_iou = float('inf')
# Training loop
for epoch in tqdm(range(num_epochs)):
    t_loss = 0.0
    deeplab_model.train()
    for batch in tqdm(data_loader):
        images = batch['image']
        masks = batch['mask']
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = deeplab_model(images)
        #masks = masks.squeeze(1)
        masks = masks.long()
        loss = criterion(outputs['out'], masks)
        t_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    deeplab_model.eval()
    total_loss = 0.0
    total_iou = 0.0
    with torch.no_grad():
        for val_batch in tqdm(valloader):
            val_images = val_batch['image'].to(device)
            masks = val_batch['mask'].to(device)
            outputs = deeplab_model(val_images)
#             masks = masks.squeeze(1)
            masks = masks.long()
            #print(torch.argmax(outputs['out'],dim=1).size(),masks.size())
            v_loss = criterion(outputs['out'], masks)
            total_iou += float(iou(torch.argmax(outputs['out'], dim=1), masks))
            total_loss += v_loss.item()
        avg_val_loss = total_loss / len(valloader.dataset)
        avg_iou = total_iou/len(valloader.dataset)
        if epoch > 20:
            if avg_iou < best_iou :
                best_iou = avg_iou
                print(f'Saving the model at Epoch [{epoch + 1}/{num_epochs}]')
                torch.save(deeplab_model.state_dict(), 'deeplab_model.pth')# will save the model in the same path 
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {t_loss/len(data_loader.dataset)}, Val Loss: {avg_val_loss}, iou:{avg_iou}')