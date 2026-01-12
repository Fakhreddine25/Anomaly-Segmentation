# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr, plot_barcode
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch.nn.functional as F
from utils import get_MSP_score, get_MaxE_score, get_MaxL_score, get_MSP_T_score

seed = 42


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)


# Function to return the anomaly_map based on passed method and passed tempScale for MSP-T
def get_anomaly_prediction(outputs, method, tempScale):
    if method == "MSP":
        score = get_MSP_score(outputs)
    elif method == "MaxL":
        score = get_MaxL_score(outputs)
    elif method == "MaxE":
        score = get_MaxE_score(outputs)
    elif method == "MSP-T":
        score = get_MSP_T_score(outputs, tempScale)
    else:
        raise ValueError(f"Uknown method: {method}")

    anomaly_map_np = score.data.cpu().numpy()

    return anomaly_map_np


def visualize_result(
    original_image_path, gt_mask, anomaly_score_map, output_filepath, method
):
    """
    Visualizes the original image, the ground truth mask, and the predicted
    anomaly score map for a single image, saving the result to a specified filepath.
    """
    try:

        orig_img = Image.open(original_image_path).convert("RGB")
        orig_img = orig_img.resize((1024, 512), Image.BILINEAR)
        orig_img = np.array(orig_img)

        plt.figure(figsize=(15, 5))

        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(orig_img)
        ax1.set_title("Original Image")
        ax1.axis("off")

        plt.subplot(1, 3, 2)
        if gt_mask is not None:
            gt_mask = np.where(gt_mask == 255, 0, gt_mask)
            plt.imshow(gt_mask, cmap=cm.get_cmap("binary").reversed())
        plt.title("Ground Truth")
        plt.axis("off")

        ax3 = plt.subplot(1, 3, 3)

        score_min = np.min(anomaly_score_map)
        score_max = np.max(anomaly_score_map)

        if "MSP" in method:
            score_min, score_max = 0, 1
            img = ax3.imshow(
                anomaly_score_map, cmap="magma", vmin=score_min, vmax=score_max
            )
        else:
            img = ax3.imshow(
                anomaly_score_map, cmap="magma", vmin=score_min, vmax=score_max
            )
        ax3.set_title(f"Predicted Anomaly Score ({method})")
        ax3.axis("off")

        plt.colorbar(img, ax=ax3, fraction=0.046, pad=0.04)

        plt.tight_layout()

        plt.savefig(output_filepath)
        plt.close()

    except Exception as e:
        print(f"Error during visualization: {e}")
        print("Ensure you have matplotlib installed: pip install matplotlib")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default=r"C:\Users\karee\Desktop\Anomaly-Project\MaskArchitectureAnomaly_CourseProject\Validation_Dataset\RoadAnomaly21\images\*.png",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="MSP",
        choices=["MSP", "MaxL", "MaxE", "MSP-T"],
        help="Anomaly scoring method (MSP, MaxL, MaxE, OR MSP-T)",
    )  # Method argument to decide the post-hoc method
    parser.add_argument(
        "--loadDir",
        default=r"C:\Users\karee\Desktop\Anomaly-Project\MaskArchitectureAnomaly_CourseProject",
    )

    parser.add_argument(
        "--loadWeights", default="/trained_models/erfnet_pretrained.pth"
    )  # Relative path to pretrained model

    parser.add_argument("--loadModel", default="/eval/erfnet.py")

    parser.add_argument(
        "--subset", default="val"
    )  # can be val or train (must have labels)

    parser.add_argument(
        "--datadir",
        default=r"C:\Users\karee\Desktop\Anomaly-Project\MaskArchitectureAnomaly_CourseProject\Validation_Dataset\RoadAnomaly21",
    )
    parser.add_argument(
        "--save_logits",
        action="store_true",
        help="If set, saves the raw logits for the case of MSP",
    )

    parser.add_argument(
        "--logits_dir",
        type=str,
        default="logits",
    )  # Saved logits directory

    parser.add_argument(
        "--tempScale",
        type=float,
        default=1.0,
        help="Temperature scaling factor for MSP method",
    )  # Temperature scale argument to decide post-processing in MSP-T

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []

    VIS_OUTPUT_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "vis_erfnet", args.method
    )
    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)
    print(f"Visualization results will be saved to: {VIS_OUTPUT_DIR}")

    if not os.path.exists("results.txt"):
        open("results.txt", "w").close()
    file = open("results.txt", "a")

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(
        model, state_dict
    ):  # custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(
        model, torch.load(weightspath, map_location=lambda storage, loc: storage)
    )
    print("Model and weights LOADED successfully")
    model.eval()
    LOGITS_OUTPUT_DIR = os.path.join(args.loadDir, args.logits_dir)
    if args.save_logits:
        if args.method == "MSP":
            os.makedirs(LOGITS_OUTPUT_DIR, exist_ok=True)
            print(f"Logits will be saved to: {LOGITS_OUTPUT_DIR}")
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        base_filename = os.path.basename(path).split(".")[0]
        logits_path = os.path.join(LOGITS_OUTPUT_DIR, f"{base_filename}.pt")
        print(path)
        images = (
            input_transform((Image.open(path).convert("RGB")))
            .unsqueeze(0)
            .float()
            .cuda()
        )

        if not os.path.exists(logits_path) and args.method == "MSP-T":

            with torch.no_grad():
                print(
                    "Logits file not found for MSP-T. Please run MSP first to generate logits."
                )

        elif os.path.exists(logits_path) and args.method == "MSP-T":

            result = torch.load(logits_path).cuda()
        else:
            with torch.no_grad():
                result = model(images)
            if args.save_logits:
                if args.method == "MSP":
                    torch.save(result.cpu(), logits_path)

        anomaly_result = get_anomaly_prediction(
            result,
            args.method,
            args.tempScale,
        )  # Get anomaly_map

        # Adapt for different image extensions
        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

        # RA: Unique labels: [0, 1, 2]  --> Remap 2 to 1
        # RA21, RO21, FS_LF, FS_Static: Unique labels: [0,1,255]  --> Remap 255 to 1
        if "RoadAnomaly" in pathGT:
            ood_gts = np.where(ood_gts == 2, 1, ood_gts)
        if "FS_LostFound_full" in pathGT or "fs_static" in pathGT:
            ood_gts = np.where(ood_gts == 255, 1, ood_gts)

        # ood_gts = ood_gts.astype(np.uint8)

        if 1 not in np.unique(ood_gts):
            continue

        else:
            if args.method in ["MaxL", "RbA", "MaxE"]:
                min_v, max_v = anomaly_result.min(), anomaly_result.max()
                if max_v != min_v:
                    anomaly_result = (anomaly_result - min_v) / (max_v - min_v)
            ood_gts_list.append(ood_gts)
            anomaly_score_list.append(anomaly_result)
            base_filename = os.path.basename(path).split(".")[0]
            viz_filename = os.path.join(VIS_OUTPUT_DIR, f"{base_filename}.png")

            visualize_result(
                original_image_path=path,
                gt_mask=ood_gts,
                anomaly_score_map=anomaly_result,
                output_filepath=viz_filename,
                method=args.method,
            )
            print(f"Saved visualization for {base_filename}")

        del result, anomaly_result, ood_gts, mask
        torch.cuda.empty_cache()
    print("Number of processed images: ", len(ood_gts_list))

    file.write("\n")

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = ood_gts == 1
    ind_mask = ood_gts == 0

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))
    valid_mask = val_label != 255
    val_out = val_out[valid_mask]
    val_label = val_label[valid_mask]
    print("val_label size: ", len(val_label))
    print("val_out size: ", len(val_out))
    print("unique labels: ", np.unique(val_label))
    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f"AUPRC score: {prc_auc*100.0}")
    print(f"FPR@TPR95: {fpr*100.0}")

    file.write(
        ("    AUPRC score:" + str(prc_auc * 100.0) + "   FPR@TPR95:" + str(fpr * 100.0))
    )
    file.close()


if __name__ == "__main__":
    main()
