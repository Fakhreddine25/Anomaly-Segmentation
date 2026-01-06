# Anomaly Segmentation Eval

In this folder you can find some functions to evaluate your model's output. It is designed to load the ERFNet checkpoint so you need to change it when evaluating the EoMT model. The main function to look for is evalAnomaly.py that produces the Anomaly Segmentation results. Other functions could be useful for extensions.

## Requirements:

It could work with the default runtime of Colab or other versions of the libraries but these are the requirements this code was tested on.

* [**Python 3.6**](https://www.python.org/): If you don't have Python3.6 in your system, I recommend installing it with [Anaconda](https://www.anaconda.com/download/#linux)
* [**PyTorch**](http://pytorch.org/): Make sure to install the Pytorch version for Python 3.6 with CUDA support (code only tested for CUDA 8.0 but it should work with higher versions).
* **Additional Python packages**: numpy, matplotlib, Pillow, torchvision and visdom (optional for --visualize flag)
* **For testing the anomaly segmentation model**: Road Anomaly, Road Obstacle, and Fishyscapes dataset. All testing images are provided here [Link](https://drive.google.com/file/d/1r2eFANvSlcUjxcerjC8l6dRa0slowMpx/view).

## Anomaly Inference:

* Anomaly Inference Command:```python evalAnomaly.py --input '/home/amarinai/segmentation/unk-dataset/RoadAnomaly21/images/*.png```. Change the dataset path ```'/home/amarinai/segmentation/unk-dataset/RoadAnomaly21/images/*.png```accordingly.

## Functions for evaluating/visualizing the network's output

Currently there are 5 usable functions to evaluate stuff:
- evalAnomaly
- eval_cityscapes_color
- eval_cityscapes_server
- eval_iou
- eval_forwardTime


## evalAnomaly.py

This code can be used to produce anomaly segmentation results on various anomaly metrics on the validation datasets you can download [here](https://drive.google.com/file/d/1zcayoIIJztxKuHOIjmSjGoQBDy4RdETr/view?usp=drive_link)

**Examples:**
```
python evalAnomaly.py --input '/home/amarinai/ViT-Adapter/segmentation/unk-dataset/RoadAnomaly21/images/*.png'
```

# Code on Citiscapes (probably not needed)

This code can be used to produce segmentation of the Cityscapes images in color for visualization purposes. By default it saves images in eval/save_color/ folder. You can also visualize results in visdom with --visualize flag.

* [**The Cityscapes dataset**](https://www.cityscapes-dataset.com/): Download the "leftImg8bit" for the RGB images and the "gtFine" for the labels. **Please note that for training you should use the "_labelTrainIds" and not the "_labelIds", you can download the [cityscapes scripts](https://github.com/mcordts/cityscapesScripts) and use the [conversor](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py) to generate trainIds from labelIds**
## eval_cityscapes_color.py 

**Options:** Specify the Cityscapes folder path with '--datadir' option. Select the cityscapes subset with '--subset' ('val', 'test', 'train' or 'demoSequence'). For other options check the bottom side of the file.

**Examples:**
```
python eval_cityscapes_color.py --datadir /home/datasets/cityscapes/ --subset val
```

## eval_cityscapes_server.py 

This code can be used to produce segmentation of the Cityscapes images and convert the output indices to the original 'labelIds' so it can be evaluated using the scripts from Cityscapes dataset (evalPixelLevelSemanticLabeling.py) or uploaded to Cityscapes test server. By default it saves images in eval/save_results/ folder.

**Options:** Specify the Cityscapes folder path with '--datadir' option. Select the cityscapes subset with '--subset' ('val', 'test', 'train' or 'demoSequence'). For other options check the bottom side of the file.

**Examples:**
```
python eval_cityscapes_server.py --datadir /home/datasets/cityscapes/ --subset val
```

## eval_iou.py 

This code can be used to calculate the IoU (mean and per-class) in a subset of images with labels available, like Cityscapes val/train sets.

**Options:** Specify the Cityscapes folder path with '--datadir' option. Select the cityscapes subset with '--subset' ('val' or 'train'). For other options check the bottom side of the file.

**Examples:**
```
python eval_iou.py --datadir /home/datasets/cityscapes/ --subset val
```

## eval_forwardTime.py
This function loads a model specified by '-m' and enters a loop to continuously estimate forward pass time (fwt) in the specified resolution. 

**Options:** Option '--width' specifies the width (default: 1024). Option '--height' specifies the height (default: 512). For other options check the bottom side of the file.

**Examples:**
```
python eval_forwardTime.py
```

**NOTE**: The pytorch code is a bit faster, but cudahalf (FP16) seems to give problems at the moment for some pytorch versions so this code only runs at FP32 (a bit slower).



