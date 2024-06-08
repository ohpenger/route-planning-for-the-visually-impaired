### Object segmentation in urban scene using resnet34
#### This project aim to develop a deep learning model to identify objects in urban street scenes. Using the Cityscapes dataset, which provides high-quality pixel-level annotations for a diverse range of objects commonly found in city environments, the model will be trained to accurately identify and segment objects such as roads, sidewalks, persons, cars, and the sky.
#### dataset: Cityscapes
#### model ResNet34
#### optimizer: Adam
#### Loss function: IOU score
#### performance: The model achieves a precision of around 58% in segmenting objects in urban street scenes. 

### To install required libraries:
#### for linux:
1. sudo apt-get install imagemagick
2. pip install -r requirements.txt
3. sudo apt-get install libmagick++-dev
4. (1)sudo vim /etc/ImageMagick-6/policy.xml  and  comment the policy.xml by change from \<policy domain="path" rights="none" pattern="@*" /> to \<!--\<policy domain="path" rights="none" pattern="@*" /> -->  
    notice: there are two lines you need to comment.  
   or  
   (2)sudo mv policy.xml /etc/ImageMagick-6/policy.xml

#### for macOS:
1. brew install imagemagick
2. pip install -r requirements.txt


#### for windows:
1. install imagemagic. check <https://zulko.github.io/moviepy/install.html>
2. pip install -r requirements.txt

### Edited Video Link:
https://faubox.rrze.uni-erlangen.de/getlink/fi2XDxAas5pd8tdh3Dvo8V/edited_video.mp4

### link to trained weights:
https://faubox.rrze.uni-erlangen.de/getlink/fi8Gr2BJugqAZDfF8LPjvi/deeplab_model_final.pth

### link to test video:
https://faubox.rrze.uni-erlangen.de/getlink/fi66Ra3Se2zFQg73kbNc7v/testvdo.mp4

### link to output video of rsu_vi script (final task output):
https://faubox.rrze.uni-erlangen.de/getlink/fi9ft66YXioUYuQxU65Wsi/sensationvdo.mp4

### To run the generate_video_walk file:
```
python3 generate_video_walk.py files/walking_video.mp4 edited_video.mp4

```

### to run training_pipeline.py file:
```
python3 training_pipeline.py /path/to/training/images /path/to/training/labels /path/to/validation/images /path/to/validation/labels /path/to/deeplab_model

```


### to run convert_masks_to_grayscale.py file:
```
python3 convert_masks_to_grayscale.py /path/to/config.json /path/to/training/labels /path/to/output/training/labels path/to/validation/labels /path/to/output/validation/labels /path/to/testing/labels /path/output/testing/labels

```
### to convert pytorch model to onnx file:
```
python3 sensation\helper\onnx_export.py --pytorch path\to\your\model --onnx path\to\your\onnx

```

