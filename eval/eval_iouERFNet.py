# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################

import numpy as np
import torch
import torch.nn.functional as F
import os
import importlib
import time

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

# Import utils to get anomaly score
from utils import get_MSP_score, get_MaxL_score, get_MaxE_score, get_MSP_T_score

# Import Anomaly from dataset
from dataset import Anomaly
from erfnet import ERFNet
from transform import Relabel, ToLabel, Colorize
from iouEval2 import iouEval, getColorEntry

# 20 classes for model evaluation
# 2 classes for binary anomaly in IoU
NUM_CLASSES_MODEL = 20
NUM_CLASSES_IOU = 2

image_transform = ToPILImage()
input_transform_anomaly = Compose(
    [
        Resize((512, 1024)),
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]), # Uncomment if your weights require it
    ]
)

# We use a base transform for initial loading; specific remapping is done in the loop
target_transform_anomaly = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
        ToLabel(),
    ]
)


def get_anomaly_prediction(outputs, method, threshold, tempScale):
    """
    Calculates the continuous score and binarizes it using a threshold (to be tuned).
    """
    if method == "MSP":
        score = get_MSP_score(outputs)
    elif method == "MaxL":
        score = get_MaxL_score(outputs)
    elif method == "MaxE":
        score = get_MaxE_score(outputs)
    elif method == "MSP-T":
        score = get_MSP_T_score(outputs, tempScale)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Binarize
    preds = (score > threshold).long()

    # Ensure 4D shape [B, 1, H, W] for the evaluator
    if preds.dim() == 2:
        preds = preds.unsqueeze(0).unsqueeze(0)
    elif preds.dim() == 3:
        preds = preds.unsqueeze(1)

    return preds


def main(args):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)

    # Model is initialized with 20 classes to match pre-trained weights
    model = ERFNet(NUM_CLASSES_MODEL)

    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(f"Skipping {name}: not in model definition.")
                    continue
            else:
                if own_state[name].shape != param.shape:
                    print(f"Skipping {name}: size mismatch.")
                    continue
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(
        model, torch.load(weightspath, map_location=lambda storage, loc: storage)
    )
    print("Model and weights LOADED successfully")

    model.eval()

    if not os.path.exists(args.datadir):
        print("Error: datadir could not be loaded")

    loader = DataLoader(
        Anomaly(args.datadir, input_transform_anomaly, target_transform_anomaly),
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # iouEval for evaluation of mIoU initialized for 2 classes: 0 (Normal) and 1 (Anomaly)
    iouEvalVal = iouEval(NUM_CLASSES_IOU)

    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if not args.cpu:
            images = images.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            outputs = model(images)

        # Get binary prediction tensor
        binary_predictions = get_anomaly_prediction(
            outputs, args.method, args.threshold, args.tempScale
        )

        # Ground-Truth remapping
        labels_np = labels.cpu().numpy()
        pathGT = filenameGt[0]

        # RA: Unique labels: [0, 1, 2]  --> Remap 2 to 1
        # RA21, RO21, FS_LF, FS_Static: Unique labels: [0,1,255]  --> Remap 255 to 1
        if "RoadAnomaly" in pathGT:
            labels_binary = np.where((labels_np == 2), 1, 0)
        elif "RoadObsticle21" in pathGT or "RoadAnomaly21" in pathGT:
            labels_binary = np.where((labels_np == 255), 1, 0)
        elif "FS_LostFound_full" in pathGT or "fs_static" in pathGT:
            labels_binary = np.where((labels_np == 255), 1, 0)

        labels = torch.from_numpy(labels_binary).long()
        if not args.cpu:
            labels = labels.cuda()

        # Ensure labels is also 4D [B, 1, H, W]
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)

        iouEvalVal.addBatch(binary_predictions, labels)

        filenameSave = os.path.basename(filename[0])
        print(step, filenameSave)

    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        val = iou_classes[i].item()
        iouStr = getColorEntry(val) + "{:0.2f}".format(val * 100) + "\033[0m"
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print(f"Results for Method: {args.method} (Threshold: {args.threshold})")
    print("Took ", time.time() - start, "seconds")
    print("=======================================")
    print("Per-Class IoU:")
    print(iou_classes_str[0], "Normal (In-Distribution)")
    print(iou_classes_str[1], "Anomaly (Out-of-Distribution)")
    print("=======================================")
    mIoU_val = iouVal.item()
    iouStr = getColorEntry(mIoU_val) + "{:0.2f}".format(mIoU_val * 100) + "\033[0m"
    print("MEAN IoU: ", iouStr, "%")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--state")
    parser.add_argument(
        "--method",
        type=str,
        default="MSP",
        choices=["MSP", "MaxL", "MaxE", "MSP-T"],
        help="Anomaly scoring method (MSP, MaxL, MaxE, or MSP-T)",
    )  # Method argument to decide the post-hoc method
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used to convert continuous score to binary prediction (0 or 1)",
    )
    parser.add_argument(
        "--loadDir",
        default=r"C:\Users\karee\Desktop\Anomaly-Project\MaskArchitectureAnomaly_CourseProject",
    )
    parser.add_argument(
        "--loadWeights", default="/trained_models/erfnet_pretrained.pth"
    )  # Relative path to pretrained model
    parser.add_argument("--loadModel", default="/eval/erfnet.py")
    parser.add_argument("--subset", default="val")
    parser.add_argument(
        "--datadir",
        default=r"C:\Users\karee\Desktop\Anomaly-Project\MaskArchitectureAnomaly_CourseProject\Validation_Dataset\RoadAnomaly21",
    )
    parser.add_argument(
        "--tempScale",
        type=float,
        default=1.0,
        help="Temperature scaling factor for MSP-T metohd",
    )  # Temperature scale argument to decide post-processing in MSP-T
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")

    main(parser.parse_args())
