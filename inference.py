import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
import torchvision.models.segmentation as segmentation
import numpy as np
import sys
from tqdm import tqdm
model_path = sys.argv[4]

model = segmentation.deeplabv3_resnet50(pretrained=False, num_classes=8)

model.load_state_dict(torch.load(model_path))

model.eval()

transform = transforms.Compose([
    transforms.Resize((512,256)),
    transforms.ToTensor(),
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image
import os

test_images_folder = sys.argv[1] #'/mapillary/Mapillary-Vistas-1000-sidewalks/testing/images' # please edit to correct location of the mapilary dataset since we have not uploded the data set
ground_truth_folder = sys.argv[2] #'/mapillary/Mapillary-Vistas-1000-sidewalks/testing/labels'
output_folder = sys.argv[3] #'/masks'

os.makedirs(output_folder, exist_ok=True)

for image_name in os.listdir(test_images_folder):
    image_path = os.path.join(test_images_folder, image_name)
    image = preprocess_image(image_path).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)

    model = model.to(device)
    with torch.no_grad():
        output = model(image)

    predicted_mask = (output['out'].argmax(1) == 1).float()
    
    #predicted_mask = torch.nn.functional.softmax(output['out'], dim=1)[0].cpu().numpy()
    #predicted_mask = (predicted_mask * 255).astype('uint8')

    output_path = os.path.join(output_folder, f"{image_name}_predicted_mask.png")
    Image.fromarray((predicted_mask.squeeze().cpu().numpy() * 255).astype('uint8')).save(output_path)
    #Image.fromarray(predicted_mask.transpose(1, 2, 0)).save(output_path)


def compute_iou(predicted_mask_path, ground_truth_mask_path):
    predicted_mask = np.array(Image.open(predicted_mask_path))
    ground_truth_mask = np.array(transform(Image.open(ground_truth_mask_path)))

    intersection = np.logical_and(predicted_mask, ground_truth_mask)
    union = np.logical_or(predicted_mask, ground_truth_mask)

    iou = np.sum(intersection) / np.sum(union)

    return iou

def evaluate_model(predictions_folder, ground_truth_folder):
    predicted_files = os.listdir(predictions_folder)
    total_iou = 0
    total_images = len(predicted_files)

    for file in tqdm(predicted_files):
        predicted_mask_path = os.path.join(predictions_folder, file)
        ground_truth_mask_path = os.path.join(ground_truth_folder, file.replace(".jpg_predicted_mask.png",".png"))

        iou = compute_iou(predicted_mask_path, ground_truth_mask_path)
        total_iou += iou

    average_iou = total_iou / total_images
    return average_iou

iou = evaluate_model(output_folder, ground_truth_folder)
print(f"Average IoU: {iou}")
