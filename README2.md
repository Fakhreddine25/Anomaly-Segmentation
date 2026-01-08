README.MD



# **Pixel vs Mask Architecture Anomaly segmentation.**



This repository makes use of a starter-code set-up for the real-time Anomaly Segmentation project of the Machine Learning Course. The project makes use of the code to implement two models: ERFNet for pixel-based segmentation and EoMT for mask-based segmentation. After comparing both approaches, temperature scaling was also used in order to mitigate the results of mask-based architecture (EoMT). 



## The code is able to reproduce the following results:

* evaluation of pixel-based architecture (ERFNet): [**evalAnomalyERFNet.py**](https://github.com/Fakhreddine25/Anomaly-Segmentation/blob/main/eval/evalAnomalyERFNet.py)
* evaluation of mask-based architecture (EoMT): [**evalAnomalyEoMT.py**](https://github.com/Fakhreddine25/Anomaly-Segmentation/blob/main/eval/evalAnomalyEoMT.py)
* evaluation of mIoU for ERFNet: [**eval_iouERFNet.py**](https://github.com/Fakhreddine25/Anomaly-Segmentation/blob/main/eval/eval_iouERFNet.py)
* evaluation of mIoU for EoMT: [**eval_iouEoMT.py**](https://github.com/Fakhreddine25/Anomaly-Segmentation/blob/main/eval/eval_iouEoMT.py)





The evalAnomaly.py code was originally designed for an ERFNet model pre-trained on cityscapes, which requires adaptation to be evaluated over our datasets.



The reason that several codes are needed for different evaluations is that different approaches require different code adaptation in order to produce their evaluation over our dataset benchmark (SMIYC - FS - RA).



## eval\_AnomalyERFNet.py



This code evaluates the ERFNet over our desired datasets. It is able to produce the following metrics: AuPRC - FPR@95 - mIoU. The code takes several arguments as inputs in order to function properly as intended.

**Examples:**
```
!python eval/evalAnomalyERFNet.py --input path/to/dataset/images/*.png --method "MSP" --datadir path/to/the/dataset/folder 
```


Arguments: 
- --input: Path to the first image within the chosen dataset with the image name indicated by '*' + .png (or the specific image extension)
- --method: A str that indicates the post-hoc method to be used. Options are: [MSP, MaxL, MaxE, MSP-T]
- --datadir: Path to the chosen dataset's folder which should include two folders: Images and Labels Masks

Additional arguments (optional):\n
- --save\_logits: If used, saves the logits of the outputs in the case of "MSP".
- --tempScale: Decides the value of temperature in order to perform Temperature Scaling, used with "MSP-T".
- --logits\_dir: This is the directory to save the logits in or could be used to load previously saved logits.











