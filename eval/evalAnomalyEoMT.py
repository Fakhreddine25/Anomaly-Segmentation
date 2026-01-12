import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EOMT_ROOT = os.path.join(PROJECT_ROOT, "eomt")
if EOMT_ROOT not in sys.path:
    sys.path.insert(0, EOMT_ROOT)

print(f"Project Root: {PROJECT_ROOT}")
print(f"Added to Path: {EOMT_ROOT}")

import glob
import torch
import math
import random
import importlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from omegaconf import OmegaConf
from argparse import ArgumentParser
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from torch.nn import functional as F
from torchvision.transforms import Compose, Resize, ToTensor

try:
    from ood_metrics import calc_metrics, fpr_at_95_tpr
    from sklearn.metrics import average_precision_score
except ImportError:
    print("Warning: ood_metrics.py not found. Metrics will be skipped")


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


# DATASET Configurations
# Defines folders structures and data extension for each dataset, as well as its original size.
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

    # Resizes ViT positional embeddings to match the image target resolution.

    assert (
        pos_embed.dim() == 3 and pos_embed.size(0) == 1
    ), f"Unexpected pos_embed shape: {pos_embed.shape}"
    _, old_n, c = pos_embed.shape
    target_h, target_w = target_hw

    old_hw = int(math.sqrt(old_n))

    if old_hw * old_hw != old_n:

        print(
            f"Warning: Source PosEmbed not square ({old_n}). Using linear interpolation."
        )

        x = pos_embed.tanspose(1, 2)
        x = F.interpolate(
            x, size=target_h * target_w, mode="linear", align_corners=False
        )
        return x.transpose(1, 2)

    # Reshape to 2D grid -> Interpolate -> Flatten
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

    # Instantiates a Python object from a config dict.
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

    # Loads the EoMT model from YAML and weights from eomt_checkpoints. If no local weights found it can download them using Hugging Face.
    print(f"Loading configuration from {config_path}...")
    cfg = OmegaConf.load(config_path)

    net_cfg = OmegaConf.to_container(cfg["model"]["init_args"]["network"], resolve=True)
    inject_config_args(net_cfg, "img_size", img_size_tuple)
    inject_config_args(net_cfg, "num_classes", num_classes)

    print(f"Building EoMT network for resolution {img_size_tuple}...")
    model = instantiate_from_cfg(net_cfg)

    # Determine weights source.
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

    # Loads weights if found
    if state_dict_path:

        print("Loading state dict...")
        ckpt = torch.load(state_dict_path, map_location="cpu", weights_only=True)
        state = ckpt.get("state_dict", ckpt)

        clean_state = {}
        for k, v in state.items():
            k_new = k
            for prefix in ["mdoel.network", "network.", "model.", "module."]:
                if k_new.startswith(prefix):
                    k_new = k_new[len(prefix) :]
                    break
            clean_state[k_new] = v

        # Resolve mismatch in Positional Embeddings.
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
                        print(
                            "Warning: Could not infer patch_size from model. Assuming 16."
                        )
                        patch_size = (16, 16)

                    target_h_grid = img_size_tuple[0] // patch_size[0]
                    target_w_grid = img_size_tuple[1] // patch_size[1]

                    if target_h_grid * target_w_grid != model_shape[1]:

                        print(
                            f"Warning: Calculated grid {target_h_grid}x{target_w_grid} != model tokens {model_shape[1]}."
                        )
                        pass

                    clean_state[pe_key] = resize_pos_embed_rectangular(
                        clean_state[pe_key], (target_h_grid, target_w_grid)
                    )

        missing, unexpected = model.load_state_dict(clean_state, strict=False)
        print(f"Weights Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    else:
        print("WARNING: Random initialization used!")

    model.to(device)
    model.eval()
    return model


def semantic_inference(model, image_tensor):

    with torch.no_grad():
        outputs = model(image_tensor)

    # Unwrap output if wrapped in list/tuple
    if (
        isinstance(outputs, (list, tuple))
        and len(outputs) == 1
        and isinstance(outputs[0], (list, tuple))
    ):

        outputs = outputs[0]

    if not isinstance(outputs, (list, tuple)) or len(outputs) != 2:
        if hasattr(model, "network"):
            outputs = model.network(image_tensor)
        else:
            raise ValueError(f"Unexpected model output format: {type(outputs)}")

    mask_logits_list, class_logits_list = outputs
    mask_logits = mask_logits_list[-1]
    class_logits = class_logits_list[-1]

    return mask_logits, class_logits


def visualize_result(
    original_image_path, gt_mask, anomaly_score_map, output_filepath, metohd, resize_dim
):
    try:
        orig_img = Image.open(original_image_path).convert("RGB")
        orig_img = orig_img.resize((resize_dim[1], resize_dim[0]), Image.BILINEAR)
        orig_img = np.array(orig_img)

        plt.figure(figsize=(15, 5))

        # Original Image
        plt.subplot(1, 3, 1)
        plt.imshow(orig_img)
        plt.title("Original Image")
        plt.axis("off")

        # Ground Truth Image
        plt.subplot(1, 3, 2)
        if gt_mask is not None:
            gt_mask = np.where(gt_mask == 255, 0, gt_mask)
            plt.imshow(gt_mask, cmap=cm.get_cmap("binary").reversed())
        plt.title("Ground Truth")
        plt.axis("off")

        # Predictions
        plt.subplot(1, 3, 3)
        score_min, score_max = anomaly_score_map.min(), anomaly_score_map.max()
        if metohd in ["MSP", "RbA", "MSP-T"]:
            score_max, score_min = 1, 0

        img = plt.imshow(
            anomaly_score_map, cmap="magma", vmin=score_min, vmax=score_max
        )
        plt.title(f"Score: {metohd}")
        plt.axis("off")
        plt.colorbar(img, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(output_filepath)
        plt.close()

    except Exception as e:
        print(f"Error visualizing {output_filepath}: {e}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", required=True, help="Root directory.")
    parser.add_argument(
        "--dataset", default="RoadAnomaly21", choices=DATASET_CONFIGS.keys()
    )
    parser.add_argument("--config", default="eomt_base_640.yaml")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--saveDir", default="./results")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--img_height", type=int, default=640)
    parser.add_argument("--img_width", type=int, default=1280)
    parser.add_argument(
        "--method",
        default="MSP",
        choices=["MSP", "MSP-T", "MaxLogit", "MaxEntropy", "RbA"],
    )
    parser.add_argument("--tempScale", type=float, default=1.0)

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

    if args.save:
        os.makedirs(args.saveDir, exist_ok=True)

    anomaly_scores_list = []
    ood_gts_list = []

    # Evaluation Loop.
    for i, path in enumerate(images):
        if i % 10 == 0:
            print(f"Processing {i}/{len(images)}: {os.path.basename(path)}")

        try:
            image = Image.open(path).convert("RGB")

            # Find eval size.
            orig_w, orig_h = image.size
            if ds_cfg.get("eval_resize"):
                EVAL_H, EVAL_W = ds_cfg["eval_resize"]
            else:
                EVAL_H, EVAL_W = orig_h, orig_w

            image_tensor = input_transform(image).unsqueeze(0).to(device)

            # --- Inference ---

            mask_logits, class_logits = semantic_inference(model, image_tensor)

            # Method: Rejected by All (RbA)
            if args.method == "RbA":
                mask_probs = mask_logits.sigmoid()
                class_probs_all = F.softmax(class_logits, dim=-1)

                inlier_probs = class_probs_all[..., :-1]

                # Rejection score.
                rej_q = 1.0 - inlier_probs.max(dim=-1).values

                # Rejection Map
                rba_map = torch.einsum("bq,bqhw->bhw", rej_q, mask_probs)

                rba_map = rba_map / (mask_probs.sum(dim=1))

                rba_map = F.interpolate(
                    rba_map.unsqueeze(1),
                    size=(EVAL_H, EVAL_W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                anomaly_map = rba_map.squeeze(0).cpu().numpy()

            # Methods: MSP - MaxLogit - MaxEntropy
            else:
                # Softmax probabilities for probability methods (not Logits)
                if args.method != "MaxLogit":

                    class_probs_all = F.softmax(class_logits, dim=-1)[..., :-1]
                    mask_probs = mask_logits.sigmoid()

                    sem_probs_id = torch.einsum(
                        "bqk, bqhw -> bkhw", class_probs_all, mask_probs
                    )

                    # Normalize
                    sem_probs_id = sem_probs_id / (
                        sem_probs_id.sum(dim=1, keepdim=True)
                    )

                    sem_probs_id = F.interpolate(
                        sem_probs_id,
                        size=(EVAL_H, EVAL_W),
                        mode="bilinear",
                        align_corners=False,
                    )

                # Maximum Softmax Probability
                if args.method == "MSP":
                    anomaly_map = (
                        (1.0 - sem_probs_id.max(dim=1).values).squeeze().cpu().numpy()
                    )

                # Maximum Entropy
                elif args.method == "MaxEntropy":
                    anomaly_map = (
                        (-(sem_probs_id * torch.log(sem_probs_id)).sum(dim=1))
                        .squeeze()
                        .cpu()
                        .numpy()
                    )

                # Maximum Logit with logits scoring not probabilities.
                elif args.method == "MaxLogit":
                    class_term = class_logits[..., :-1]
                    mask_probs = mask_logits.sigmoid()

                    sem_map = torch.einsum("bqk, bqhw -> bkhw", class_term, mask_probs)
                    sem_map = F.interpolate(
                        sem_map,
                        size=(EVAL_H, EVAL_W),
                        mode="bilinear",
                        align_corners=False,
                    )
                    anomaly_map = (-sem_map.max(dim=1).values).squeeze().cpu().numpy()

                # MSP with Temperature scaling.
                # Requires redefinition of softmax probabilities with scaled class_logits
                elif args.method == "MSP-T":
                    t = (torch.ones(1) * args.tempScale).to(device)
                    class_probs_all = F.softmax(class_logits / t, dim=-1)[..., :-1]
                    mask_probs = mask_logits.sigmoid()
                    sem_probs_id = torch.einsum(
                        "bqk, bqhw -> bkhw", class_probs_all, mask_probs
                    )

                    sem_probs_id = sem_probs_id / (
                        sem_probs_id.sum(dim=1, keepdim=True)
                    )

                    sem_probs_id = F.interpolate(
                        sem_probs_id,
                        size=(EVAL_H, EVAL_W),
                        mode="bilinear",
                        align_corners=False,
                    )
                    anomaly_map = (
                        (1.0 - sem_probs_id.max(dim=1).values).squeeze().cpu().numpy()
                    )

        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

        # --- GT Loading ---
        filename = os.path.basename(path)
        gt_path = os.path.join(args.input, ds_cfg["target_folder"], filename)
        if not os.path.exists(gt_path):
            gt_path = gt_path.replace(ds_cfg["suffix"], ".png")
        if not os.path.exists(gt_path) and filename.endswith(".jpg"):
            gt_path = gt_path.replace(".jpg", ".png")

        if os.path.exists(gt_path):
            gt_img = Image.open(gt_path)
            if gt_img.size != (EVAL_W, EVAL_H):
                gt_img = gt_img.resize((EVAL_W, EVAL_H), Image.NEAREST)
            gt_np = np.array(gt_img)

            # Map datasets to binary combinations.
            if "RoadAnomaly" in args.dataset:
                gt_np = np.where(gt_np == 2, 1, gt_np)
            if "FS_LostFound_full" in args.dataset or "fs_static" in args.dataset:
                gt_np = np.where(gt_np == 255, 1, gt_np)

            binary_gt = gt_np.astype(np.uint8)

            # Skip images with no valid pixels
            if 1 not in np.unique(binary_gt):
                continue

            if args.method in ["MaxLogit", "RbA", "MaxEntropy", "MSP", "MSP-T"]:
                min_v, max_v = anomaly_map.min(), anomaly_map.max()
                if max_v != min_v:
                    anomaly_map = (anomaly_map - min_v) / (max_v - min_v)

            ood_gts_list.append(binary_gt.flatten())
            anomaly_scores_list.append(anomaly_map.flatten())

            if args.save:
                save_path = os.path.join(
                    args.saveDir,
                    filename.replace(ds_cfg["suffix"], f"_{args.method}.png"),
                )
                visualize_result(
                    path,
                    binary_gt,
                    anomaly_map,
                    save_path,
                    args.method,
                    (EVAL_H, EVAL_W),
                )
        else:
            if i == 0:
                print(f"DEBUG: GT missing at {gt_path}")

    # --- Metrics ---
    if ood_gts_list:
        print("\nCalculating Metrics...")
        all_scores = np.concatenate(anomaly_scores_list)
        all_gts = np.concatenate(ood_gts_list)

        # RAM optimization
        if len(all_gts) > 20_000_000:
            print(
                f"Dataset too large for RAM ({len(all_gts)} pixels). Subsampling 10%..."
            )
            all_scores = all_scores[::10]
            all_gts = all_gts[::10]

        valid_mask = all_gts != 255
        all_scores = all_scores[valid_mask]
        all_gts = all_gts[valid_mask]

        fpr = fpr_at_95_tpr(all_scores, all_gts)

        fpr = fpr_at_95_tpr(all_scores, all_gts)
        auprc = average_precision_score(all_gts, all_scores)

        print(f"Method: {args.method}")

        print(f"AUPRC: {auprc * 100:.4f}")
        print(f"FPR@95: {fpr * 100:.4f}")

        with open("results.txt", "a") as f:
            f.write(
                f"\n{args.dataset} | {args.method}: AUPRC={auprc:.4f}, FPR={fpr:.4f}"
            )
    else:
        print("No valid GT data found for metrics.")


if __name__ == "__main__":
    main()
