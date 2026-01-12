import os
import sys
import glob
import torch
import math
import random
import importlib
import numpy as np
import time
from argparse import ArgumentParser
from PIL import Image
from omegaconf import OmegaConf
from torch.nn import functional as F
from torchvision.transforms import Compose, Resize, ToTensor
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

try:
    from iouEval2 import iouEval, getColorEntry
except ImportError:
    print("Could not import iouEval2. Please ensure iouEval2.py is in the directory.")
    sys.exit(1)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EOMT_ROOT = os.path.join(PROJECT_ROOT, "eomt")
if EOMT_ROOT not in sys.path:
    sys.path.insert(0, EOMT_ROOT)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# DATASET Configurations
DATASET_CONFIGS = {
    "RoadAnomaly21": {
        "img_folder": "images",
        "target_folder": "labels_masks",
        "suffix": ".png",
        "eval_resize": None,
    },
    "RoadObsticle21": {
        "img_folder": "images",
        "target_folder": "labels_masks",
        "suffix": ".webp",
        "eval_resize": None,
    },
    "fs_static": {
        "img_folder": "images",
        "target_folder": "labels_masks",
        "suffix": ".jpg",
        "eval_resize": (1024, 2048),
    },
    "FS_LostFound_full": {
        "img_folder": "images",
        "target_folder": "labels_masks",
        "suffix": ".png",
        "eval_resize": (1024, 2048),
    },
    "RoadAnomaly": {
        "img_folder": "images",
        "target_folder": "labels_masks",
        "suffix": ".jpg",
        "eval_resize": None,
    },
}


def resize_pos_embed_rectangular(pos_embed, target_hw):
    assert pos_embed.dim() == 3 and pos_embed.size(0) == 1
    _, old_n, c = pos_embed.shape
    target_h, target_w = target_hw
    old_hw = int(math.sqrt(old_n))

    if old_hw * old_hw != old_n:
        x = pos_embed.transpose(1, 2)
        x = F.interpolate(
            x, size=target_h * target_w, mode="linear", align_corners=False
        )
        return x.transpose(1, 2)

    x = pos_embed.reshape(1, old_hw, old_hw, c).permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=False)
    x = x.permute(0, 2, 3, 1).reshape(1, target_h * target_w, c)
    return x


def inject_config_args(node, key, value):
    if isinstance(node, dict):
        if "class_path" in node:
            cls_name = node["class_path"].split(".")[-1]
            if (key == "num_classes" and cls_name == "EoMT") or (
                key == "img_size" and cls_name == "ViT"
            ):
                init_args = node.get("init_args", {})
                if init_args is None:
                    init_args = {}
                init_args[key] = value
                node["init_args"] = init_args
        for v in node.values():
            inject_config_args(v, key, value)
    elif isinstance(node, list):
        for v in node:
            inject_config_args(v, key, value)


def instantiate_from_cfg(node):
    if isinstance(node, dict) and "class_path" in node:
        class_path = node["class_path"]
        init_args = node.get("init_args", {}) or {}
        init_args = {k: instantiate_from_cfg(v) for k, v in init_args.items()}
        module_name, cls_name = class_path.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), cls_name)
        return cls(**init_args)
    if isinstance(node, dict):
        return {k: instantiate_from_cfg(v) for k, v in node.items()}
    if isinstance(node, list):
        return [instantiate_from_cfg(v) for v in node]
    return node


def load_eomt_model(
    config_path, checkpoint_path, device, img_size_tuple, num_classes=19
):
    print(f"Loading configuration from {config_path}...")
    cfg = OmegaConf.load(config_path)
    net_cfg = OmegaConf.to_container(cfg["model"]["init_args"]["network"], resolve=True)
    inject_config_args(net_cfg, "img_size", img_size_tuple)
    inject_config_args(net_cfg, "num_classes", num_classes)

    print(f"Building EoMT Network for resolution {img_size_tuple}...")
    model = instantiate_from_cfg(net_cfg)

    state_dict_path = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Using local checkpoint: {checkpoint_path}")
        state_dict_path = checkpoint_path
    else:
        name = cfg.get("trainer", {}).get("logger", {}).get("init_args", {}).get("name")
        if name:
            try:
                state_dict_path = hf_hub_download(
                    repo_id=f"tue-mps/{name}", filename="pytorch_model.bin"
                )
            except RepositoryNotFoundError:
                print(f"Warning: No HF weights found for {name}")

    if state_dict_path:
        print("Loading state dict...")
        ckpt = torch.load(state_dict_path, map_location="cpu", weights_only=True)
        state = ckpt.get("state_dict", ckpt)
        clean_state = {}
        for k, v in state.items():
            k_new = k
            for prefix in ["model.network.", "network.", "model.", "module."]:
                if k_new.startswith(prefix):
                    k_new = k_new[len(prefix) :]
                    break
            clean_state[k_new] = v

        pe_key = "encoder.backbone.pos_embed"
        if pe_key in clean_state:
            model_sd = model.state_dict()
            if pe_key in model_sd:
                ckpt_shape = clean_state[pe_key].shape
                model_shape = model_sd[pe_key].shape
                if ckpt_shape != model_shape:
                    print(
                        f"Resizing pos_embed: ckpt {ckpt_shape} -> model {model_shape}"
                    )
                    try:
                        patch_size = model.encoder.backbone.patch_embed.patch_size
                        if isinstance(patch_size, int):
                            patch_size = (patch_size, patch_size)
                    except AttributeError:
                        patch_size = (16, 16)
                    target_h_grid = img_size_tuple[0] // patch_size[0]
                    target_w_grid = img_size_tuple[1] // patch_size[1]
                    clean_state[pe_key] = resize_pos_embed_rectangular(
                        clean_state[pe_key], (target_h_grid, target_w_grid)
                    )

        model.load_state_dict(clean_state, strict=False)
        print("Weights Loaded.")
    else:
        print("WARNING: Random initialization used!")

    model.to(device)
    model.eval()
    return model


def semantic_inference(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)

    if (
        isinstance(outputs, (list, tuple))
        and len(outputs) == 1
        and isinstance(outputs[0], (list, tuple))
    ):
        outputs = outputs[0]
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 2:
        if hasattr(model, "network"):
            outputs = model.network(image_tensor)

    mask_logits_list, class_logits_list = outputs
    mask_logits = mask_logits_list[-1]
    class_logits = class_logits_list[-1]
    return mask_logits, class_logits


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="Root directory containing datasets"
    )
    parser.add_argument(
        "--dataset", default="RaoadAnomaly21", choices=DATASET_CONFIGS.keys()
    )
    parser.add_argument("--config", default="eomt_base_640.yaml")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--img_height", type=int, default=640)
    parser.add_argument("--img_width", type=int, default=1280)
    parser.add_argument(
        "--method",
        default="MSP",
        choices=["MSP", "MSP-T", "MaxLogit", "MaxEntropy", "RbA"],
    )

    # Threshold used to binarize the score distribution.
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold to binarize anomaly score",
    )
    parser.add_argument(
        "--tempScale",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()
    device = torch.device("cpu" if args.cpu else "cuda")
    ds_cfg = DATASET_CONFIGS[args.dataset]

    image_dir = os.path.join(args.input, ds_cfg["img_folder"])
    if (
        not os.path.exists(image_dir)
        and os.path.basename(os.path.normpath(args.input)) == ds_cfg["img_folder"]
    ):
        image_dir = args.input
        args.input = os.path.dirname(args.input)

    print(f"Scanning images in: {image_dir}")
    images = sorted(
        glob.glob(os.path.join(image_dir, "**", f"*{ds_cfg['suffix']}"), recursive=True)
    )
    if not images:
        images = sorted(
            glob.glob(os.path.join(image_dir, "**", "*.png"), recursive=True)
        )
    print(f"Found {len(images)} images.")

    img_size_tuple = (args.img_height, args.img_width)
    try:
        model = load_eomt_model(args.config, args.ckpt, device, img_size_tuple)
    except Exception as e:
        print(f"Critical Error loading model: {e}")
        return

    input_transform = Compose(
        [
            Resize(img_size_tuple, Image.BILINEAR),
            ToTensor(),
        ]
    )

    if args.dataset in ["RoadAnomaly21", "RoadObsticle21"]:
        iouEvalVal = iouEval(nClasses=3, ignoreIndex=2)
    else:
        iouEvalVal = iouEval(nClasses=2)

    start = time.time()

    for i, path in enumerate(images):
        if i % 10 == 0:
            print(f"Processing {i}/{len(images)}: {os.path.basename(path)}")

        try:
            image = Image.open(path).convert("RGB")
            orig_w, orig_h = image.size
            if ds_cfg.get("eval_resize"):
                EVAL_H, EVAL_W = ds_cfg["eval_resize"]
            else:
                EVAL_H, EVAL_W = orig_h, orig_w

            image_tensor = input_transform(image).unsqueeze(0).to(device)

            mask_logits, class_logits = semantic_inference(model, image_tensor)

            if args.method == "RbA":
                mask_probs = mask_logits.sigmoid()

                class_probs_all = F.softmax(class_logits, dim=-1)

                inlier_probs = class_probs_all[..., :-1]

                rej_q = 1.0 - inlier_probs.max(dim=-1).values

                rba_map = torch.einsum("bq,bqhw->bhw", rej_q, mask_probs)

                rba_map = rba_map / (mask_probs.sum(dim=1) + 1.0)

                rba_map = F.interpolate(
                    rba_map.unsqueeze(1),
                    size=(EVAL_H, EVAL_W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                anomaly_map = rba_map.squeeze(0).cpu().numpy()

            elif args.method == "MaxLogit":
                mask_probs = mask_logits.sigmoid()

                class_term = class_logits[..., :-1]

                pixel_logits = torch.einsum("bqk, bqhw -> bkhw", class_term, mask_probs)

                pixel_logits = F.interpolate(
                    pixel_logits,
                    size=(EVAL_H, EVAL_W),
                    mode="bilinear",
                    align_corners=False,
                )

                anomaly_map = (-pixel_logits.max(dim=1).values).squeeze().cpu().numpy()

            else:
                mask_probs = mask_logits.sigmoid()

                if args.method == "MSP-T":
                    t = (torch.ones(1) * args.tempScale).to(device)
                    class_probs = F.softmax(class_logits / t, dim=-1)[..., :-1]
                else:

                    class_probs = F.softmax(class_logits, dim=-1)[..., :-1]

                sem_map = torch.einsum("bqk, bqhw -> bkhw", class_probs, mask_probs)

                sem_map = sem_map / (sem_map.sum(dim=1, keepdim=True) + 1e-6)

                sem_map = F.interpolate(
                    sem_map,
                    size=(EVAL_H, EVAL_W),
                    mode="bilinear",
                    align_corners=False,
                )

                if args.method in ["MSP", "MSP-T"]:
                    anomaly_map = (
                        (1.0 - sem_map.max(dim=1).values).squeeze().cpu().numpy()
                    )

                elif args.method == "MaxEntropy":
                    log_probs = torch.log(sem_map + 1e-12)
                    entropy = -(sem_map * log_probs).sum(dim=1)
                    max_ent = math.log(sem_map.shape[1])
                    anomaly_map = (entropy / max_ent).squeeze().cpu().numpy()

            filename = os.path.basename(path)
            gt_path = os.path.join(args.input, ds_cfg["target_folder"], filename)

            if not os.path.exists(gt_path):
                gt_path = gt_path.replace(ds_cfg["suffix"], ".png")
            if not os.path.exists(gt_path) and filename.endswith(".jpg"):
                gt_path = gt_path.replace(".jpg", ".png")
            if not os.path.exists(gt_path) and ds_cfg["suffix"] == ".webp":
                gt_path = gt_path.replace(".png", ".webp")

            if os.path.exists(gt_path):
                gt_img = Image.open(gt_path)
                if gt_img.size != (EVAL_W, EVAL_H):
                    gt_img = gt_img.resize((EVAL_W, EVAL_H), Image.NEAREST)
                gt_np = np.array(gt_img)

                if "RoadAnomaly" in args.dataset:
                    gt_np = np.where(gt_np == 2, 1, gt_np)
                if "FS_LostFound_full" in args.dataset or "fs_static" in args.dataset:
                    gt_np = np.where(gt_np == 255, 1, gt_np)
                if "RoadAnomaly21" in args.dataset or "RoadObsticle21" in args.dataset:
                    gt_np = np.where(gt_np == 255, 2, gt_np)

                binary_gt = gt_np.astype(np.uint8)

                if args.method in ["MaxLogit", "RbA", "MaxEntropy"]:
                    min_v, max_v = anomaly_map.min(), anomaly_map.max()
                    if max_v != min_v:
                        anomaly_map = (anomaly_map - min_v) / (max_v - min_v)
                pred_mask = (anomaly_map > args.threshold).astype(np.uint8)

                pred_tensor = (
                    torch.from_numpy(pred_mask)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .long()
                    .to(device)
                )
                gt_tensor = (
                    torch.from_numpy(binary_gt)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .long()
                    .to(device)
                )

                iouEvalVal.addBatch(pred_tensor, gt_tensor)

            else:
                print(f"Warning: GT not found for {filename}")

        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = (
            getColorEntry(iou_classes[i])
            + "{:0.2f}".format(iou_classes[i] * 100)
            + "\033[0m"
        )
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print("Took ", time.time() - start, "seconds")
    print("=======================================")
    print(f"Dataset: {args.dataset}")
    print(f"Method: {args.method} (Threshold: {args.threshold})")
    print("Per-Class IoU:")
    print(iou_classes_str[0], "Normal/Background")
    print(iou_classes_str[1], "Anomaly")
    print("=======================================")
    iouStr = getColorEntry(iouVal) + "{:0.2f}".format(iouVal * 100) + "\033[0m"
    print("MEAN IoU: ", iouStr, "%")


if __name__ == "__main__":
    main()
