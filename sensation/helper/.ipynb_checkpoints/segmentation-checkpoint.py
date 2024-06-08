import torch.nn as nn
import segmentation_models_pytorch as smp
import torchvision.models.segmentation as segmentation
import torch
import numpy as np

class DeepLabV3PResNet50(nn.Module):
    def __init__(self, num_classes: int = 8):
        super(DeepLabV3PResNet50, self).__init__()
        self.layer = segmentation.deeplabv3_resnet50(pretrained=False, num_classes=8)
        self.activation = nn.Sigmoid()
        self.n_classes = num_classes

    def forward(self, x):
        return self.layer(x)
    

class Segmentator():
    def __init__(self,model):
        self.model = model
    
    def mask_to_rgb(self, mask):
        colormap = {(128, 64, 128): 1,
                    (244, 35, 232): 2,
                    (220, 20, 60): 5,
                    (250, 170, 30): 3,
                    (192, 192, 192): 4,
                    (119, 11, 32): 7,
                    (0, 0, 142): 6,
                    (0, 0, 0): 0}
        mask = np.array(mask)
        rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)

        for rgb_val,label in colormap.items():
            rgb_mask[mask==label] = rgb_val
        return rgb_mask

    def inference(self,image):
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        image = image.to(device)

        self.model = self.model.to(device)
        with torch.no_grad():
            output = self.model(image)

        predicted_mask = (output['out'].argmax(1)).float()

        return predicted_mask