import torch.nn as nn
import segmentation_models_pytorch as smp
import torchvision.models.segmentation as segmentation
import torch
import numpy as np
import onnxruntime
import cv2
from PIL import Image
from torchvision import transforms

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
        self.model_path = model
    
    def mask_to_rgb(self,mask):
        colormap = {(128, 64, 128): 1,
                    (244, 35, 232): 2,
                    (220, 20, 60): 5,
                    (250, 170, 30): 3,
                    (192, 192, 192): 4,
                    (119, 11, 32): 7,
                    (0, 0, 142): 6,
                    (0, 0, 0): 0}
        
        mask = np.array(mask)
        rgb_mask = np.zeros((mask.shape[0],mask.shape[1], 3), dtype=np.uint8)
        for rgb_val,label in colormap.items():
            rgb_mask[np.all(mask==label, axis=-1)] = rgb_val
        return rgb_mask

    def inference(self,frame):
        transform = transforms.ToTensor()
        onnx_model_path = './model_weights/resnet50_imagenet.onnx'
        ort_session = onnxruntime.InferenceSession(onnx_model_path)

        # Print the input and output names of the model
        input_names = ort_session.get_inputs()
        output_names = ort_session.get_outputs()
        #print("Input names:", [input_.name for input_ in input_names])
        #print("Output names:", [output.name for output in output_names])
        input_shape = input_names[0].shape

        input_image = cv2.resize(frame, (input_shape[3], input_shape[2]))
        input_image = transform(input_image).numpy()

        # Convert the input image to the format expected by ONNX runtime (float32)
        input_image = input_image.astype(np.float32)

        # Add batch dimension if the model expects batched input
        if len(input_shape) == 4:
            input_image = np.expand_dims(input_image, axis=0)

        # Perform inference
        output = ort_session.run(None, {input_names[0].name: input_image})
        mask = output[0].argmax(1)
        mask = np.transpose(mask, (1, 2, 0))
        return mask