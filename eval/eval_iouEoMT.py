# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################

# Code to calculate IoU (mean and per-class) for EoMT Anomaly Detection
# Adapted for Mask Transformer architectures 2025
#######################
import os
import sys

current_script_path = os.path.abspath(__file__)
eval_folder = os.path.dirname(current_script_path)
project_root = os.path.abspath(os.path.join(eval_folder, ".."))
eomt_folder = os.path.join(project_root, "eomt")

for path in [project_root, eomt_folder]:
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
import torch.nn.functional as F
import importlib
import time
import glob
from PIL import Image
from argparse import ArgumentParser
from omegaconf import OmegaConf
from typing import Any
from dataset import Anomaly

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage

# Project Modules
try:
    from ood_metrics import fpr_at_95_tpr
except ImportError:
    pass

try:
    import models
except ImportError:
    print(f"\n[!] CRITICAL ERROR: Could not find the 'models' folder.")
    print(f"Expected location: {os.path.join(eomt_folder, 'models')}")
    print(f"Please ensure {eomt_folder} exists and contains a 'models' subfolder.")
    sys.exit(1)

from transform import ToLabel
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


def resize_pos_embed_path_only(
    pos_embed: torch.Tensor, new_tokens: int
) -> torch.Tensor:
    assert pos_embed.dim() == 3 and pos_embed.size(0) == 1
    _, old_n, c = pos_embed.shape
    old_hw = int(old_n**0.5)
    new_hw = int(new_tokens**0.5)
    x = pos_embed.reshape(1, old_hw, old_hw, c).permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(new_hw, new_hw), mode="bilinear", align_corners=False)
    x = x.permute(0, 2, 3, 1).reshape(1, new_hw * new_hw, c)
    return x


def instantiate_from_cfg(node: Any) -> Any:
    if isinstance(node, dict) and "class_path" in node:
        class_path = node["class_path"]
        init_args = node.get("init_args", {}) or {}
        init_args = {k: instantiate_from_cfg(v) for k, v in init_args.items()}
        (
            module_name,
            cls_name,
        ) = class_path.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), cls_name)
        return cls(**init_args)
    if isinstance(node, dict):
        return {k: instantiate_from_cfg(v) for k, v in node.items()}
    if isinstance(node, list):
        return [instantiate_from_cfg(v) for v in node]
    return node


def inject_cfg_values(node: Any, img_size: tuple, num_classes: int) -> None:
    if isinstance(node, dict):
        cp = node.get("class_path", "")
        if cp.endswith("ViT"):
            node.setdefault("init_args", {})["img_size"] = img_size
        if cp.endswith("EoMT"):
            node.setdefault("init_args", {})["num_classes"] = num_classes
        for v in node.values():
            inject_cfg_values(v, img_size, num_classes)
    elif isinstance(node, list):
        for v in node:
            inject_cfg_values(v, img_size, num_classes)


def get_anomaly_prediction(
    mask_logits, class_logits, method, threshold, tempScale, target_size=(512, 1024)
):
    """
    Calculates continuous anomaly scores for EoMT and binarizes based on threshold.
    """

    # If MSP, normalize the class_logits by temperature scale
    if method == "MSP-T":
        t = (torch.ones(1) * tempScale).cuda()
        class_probs_all = F.softmax(class_logits / t, dim=-1)[..., :-1]
        mask_probs = mask_logits.sigmoid()

    # Else, continue
    else:
        class_probs_all = F.softmax(class_logits, dim=-1)[..., :-1]
        mask_probs = mask_logits.sigmoid()

    # Construct semantic map including the VOID channel
    sem_probs_all = torch.einsum("bqk, bqhw -> bkhw", class_probs_all, mask_probs)
    # Normalize spatially across all class possibilities
    sem_probs_all = sem_probs_all / (sem_probs_all.sum(dim=1, keepdim=True) + 1e-6)
    # Resize to target resolution
    sem_probs_all = F.interpolate(
        sem_probs_all,
        size=(512, 1024),
        mode="bilinear",
        align_corners=False,
    )
    # In-Distribution probs
    sem_probs_id = sem_probs_all[:, :-1, :, :]

    if method == "MSP-T":
        anomaly_result = (1.0 - sem_probs_id.max(dim=1).values).squeeze().cpu()
    elif method == "RbA":
        mask_probs = mask_logits.sigmoid()
        class_probs_all = F.softmax(class_logits, dim=-1)  # [B, Q, C+1]
        inlier_probs = class_probs_all[..., :-1]  # [B, Q, C]
        rej_q = 1.0 - inlier_probs.max(dim=-1).values  # [B, Q]
        # pixel score: queries' rejection weighted by their mask
        rba_map = torch.einsum("bq,bqhw->bhw", rej_q, mask_probs)  # [B, Hm, Wm]
        rba_map = rba_map / (mask_probs.sum(dim=1) + 1e-6)  # [B, Hm, Wm]
        # resize to GT size (our eval size)
        rba_map = F.interpolate(
            rba_map.unsqueeze(1),
            size=(512, 1024),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        anomaly_result = rba_map.squeeze(0).cpu()  # [H, W]
    # Standard methods (MSP, MaxL, MaxE) work on in-distribution    channels
    elif method == "MSP":
        anomaly_result = (1.0 - sem_probs_id.max(dim=1).values).squeeze().cpu()
    elif method == "MaxL":
        pixel_logits = torch.einsum(
            "bqk, bqhw -> bkhw", class_logits[:, :, :-1], mask_probs
        )
        pixel_logits = F.interpolate(
            pixel_logits,
            size=(512, 1024),
            mode="bilinear",
            align_corners=False,
        )
        anomaly_result = (-pixel_logits.max(dim=1).values).squeeze().cpu()
    elif method == "MaxE":
        # Shannon Entropy
        anomaly_result = (
            (-(sem_probs_id * torch.log(sem_probs_id + 1e-12)).sum(dim=1))
            .squeeze()
            .cpu()
        )

    preds = (anomaly_result > threshold).long()

    if preds.dim() == 2:
        preds = preds.unsqueeze(0).unsqueeze(0)
    elif preds.dim() == 3:
        preds = preds.unsqueeze(1)

    return preds


def main(args):
    device = torch.device("cpu" if args.cpu else "cuda")

    # Load config file & build model
    cfg_path = os.path.normpath(os.path.join(project_root, args.eomt_config))
    if not os.path.exists(cfg_path):
        print(f"Error: Config file not found at {cfg_path}")
        return

    cfg = OmegaConf.load(cfg_path)

    data_args = cfg.get("data", {}).get("init_args", {})
    img_size_cfg = data_args.get("img_size", 640)
    num_classes = data_args.get("num_classes", 19)
    img_size_vit = (
        (img_size_cfg, img_size_cfg)
        if isinstance(img_size_cfg, int)
        else tuple(img_size_cfg)
    )

    net_node = OmegaConf.to_container(
        cfg["model"]["init_args"]["network"], resolve=True
    )
    inject_cfg_values(net_node, img_size_vit, num_classes)

    print("Instantiating model architecture...")
    model = instantiate_from_cfg(net_node)

    ckpt_path = os.path.normpath(os.path.join(project_root, args.eomt_ckpt))
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    # Load weights
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))

    def strip_prefix(k):
        for p in ("model.network.", "network.", "model.", "module."):
            if k.startswith(p):
                return k[len(p) :]
        return k

    clean_state = {strip_prefix(k): v for k, v in state.items()}
    model_sd = model.state_dict()
    pe_key = "encoder.backbone.pos_embed"
    if pe_key in clean_state and pe_key in model_sd:
        if clean_state[pe_key].shape != model_sd[pe_key].shape:
            clean_state[pe_key] = resize_pos_embed_path_only(
                clean_state[pe_key], model_sd[pe_key].shape[1]
            )

    model.load_state_dict(clean_state, strict=False)
    model.to(device).eval()

    input_transform = Compose([Resize(img_size_vit, Image.BILINEAR), ToTensor()])
    target_transform = Compose([Resize((512, 1024), Image.NEAREST), ToLabel()])

    dataset = Anomaly(args.datadir, input_transform, target_transform)
    loader = DataLoader(
        dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False
    )

    iouEvalVal = iouEval(2)
    start = time.time()

    print(
        f"Starting IoU Evaluation with method: {args.method} (Threshold: {args.threshold})"
    )

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            mask_logits_layers, class_logits_layers = model(images)
            mask_logits = mask_logits_layers[-1]
            class_logits = class_logits_layers[-1]

        # Get binary predictions
        binary_predictions = get_anomaly_prediction(
            mask_logits, class_logits, args.method, args.threshold, args.tempScale
        )

        # Ground_Truth remapping
        labels_np = labels.cpu()
        pathGT = filenameGt[0]

        # RA: Unique labels: [0, 1, 2]  --> Remap 2 to 1
        # RA21, RO21, FS_LF, FS_Static: Unique labels: [0,1,255]  --> Remap 255 to 1
        if "RoadAnomaly" in pathGT:
            labels_binary = np.where((labels_np == 2), 1, 0)
        elif "RoadObsticle21" in pathGT or "RoadAnomaly21" in pathGT:
            labels_binary = np.where((labels_np == 255), 1, 0)
        elif "FS_LostFound_full" in pathGT or "fs_static" in pathGT:
            labels_binary = np.where((labels_np == 255), 1, 0)
        else:
            labels_binary = np.where(labels_np > 0, 1, 0)

        labels_tensor = torch.from_numpy(labels_binary).long().to(device)
        if labels_tensor.dim() == 3:
            labels_tensor = labels_tensor.unsqueeze(1)

        iouEvalVal.addBatch(binary_predictions, labels_tensor)
        if (step + 1) % 5 == 0:
            print(f"[{step+1}/{len(loader)}] {os.path.basename(filename[0])}")

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
        choices=["MSP", "MaxL", "MaxE", "RbA", "MSP-T"],
        help="Anomaly scoring method (MSP, MaxL, MaxE, RbA, or MSP-T)",
    )  # Method argument to decide the post-hoc method
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used to convert continuous score to binary prediction (0 or 1)",
    )
    parser.add_argument(
        "--eomt_config",
        default="eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml",
    )  # Relative path to config file
    parser.add_argument(
        "--eomt_ckpt",
        default=r"C:\Users\karee\Desktop\Anomaly-Project\MaskArchitectureAnomaly_CourseProject\eomt_checkpoints\eomt_cityscapes.bin",
    )  # Relative path to pretrained model
    parser.add_argument(
        "--loadDir",
        default=r"C:\Users\karee\Desktop\Anomaly-Project\MaskArchitectureAnomaly_CourseProject",
    )
    parser.add_argument("--subset", default="val")
    parser.add_argument(
        "--datadir",
        default=r"C:\Users\karee\Desktop\Anomaly-Project\MaskArchitectureAnomaly_CourseProject\Validation_Dataset\RoadAnomaly21",
    )
    parser.add_argument(
        "--tempScale",
        type=float,
        default=0.5,
        help="Temperature scaling factor for MSP-T metohd",
    )  # Temperature scale argument to decide post-processing in MSP-T
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")

    main(parser.parse_args())
