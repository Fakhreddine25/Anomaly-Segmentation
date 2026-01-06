import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EOMT_ROOT = os.path.join(PROJECT_ROOT, "eomt")
if EOMT_ROOT not in sys.path:
    sys.path.insert(0, EOMT_ROOT)
import glob
import torch
import torch.nn.functional as F
import random
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
import importlib
from typing import Any
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor
import matplotlib.pyplot as plt
from collections import Counter

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)


def visualize_result(
    original_image_path, gt_mask, anomaly_score_map, output_filepath, method
):

    try:
        orig_img = Image.open(original_image_path).convert("RGB")
        orig_img = orig_img.resize((1024, 512), Image.BILINEAR)
        orig_img = np.array(orig_img)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(orig_img)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask, cmap="binary")
        plt.title("Ground Truth Anomaly")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        score_min = np.min(anomaly_score_map)
        score_max = np.max(anomaly_score_map)

        if "MSP" in method:
            score_min, score_max = 0, 1
            img = plt.imshow(
                anomaly_score_map, cmap="magma", vmin=score_min, vmax=score_max
            )
        else:
            img = plt.imshow(
                anomaly_score_map,
                cmap="magma",
                vmin=score_min,
                vmax=score_max,
            )
        plt.title(f"Score")
        plt.axis("off")
        plt.colorbar(img, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(output_filepath)
        plt.close()

    except Exception as e:
        print(f"Error during visualization of {output_filepath}: e")


def resize_pos_embed_patch_only(
    pos_embed: torch.Tensor, new_tokens: int
) -> torch.Tensor:
    """
    pos_embed is patch-only: [1, N, C] with N = H*W (no cls/reg tokens inside).
    Resizes from sqrt(N) x sqrt(N) -> sqrt(new_tokens) x sqrt(new_tokens).
    """
    assert (
        pos_embed.dim() == 3 and pos_embed.size(0) == 1
    ), f"Unexpected pos_embed shape: {pos_embed.shape}"
    _, old_n, c = pos_embed.shape

    old_hw = int(old_n**0.5)
    new_hw = int(new_tokens**0.5)

    if old_hw * old_hw != old_n or new_hw * new_hw != new_tokens:
        raise ValueError(
            f"pos_embed tokens not perfect squares: old={old_n}, new={new_tokens}"
        )

    x = pos_embed.reshape(1, old_hw, old_hw, c).permute(0, 3, 1, 2)  # [1,C,H,W]
    x = F.interpolate(x, size=(new_hw, new_hw), mode="bilinear", align_corners=False)
    x = x.permute(0, 2, 3, 1).reshape(1, new_hw * new_hw, c)  # [1,newN,C]
    return x


def inject_num_classes_into_eomt(node: Any, num_classes_value: int) -> None:
    if isinstance(node, dict):
        class_path = node.get("class_path", "")
        if class_path.split(".")[-1] == "EoMT":
            init_args = node.get("init_args", None)
            if not isinstance(init_args, dict):
                init_args = {}
                node["init_args"] = init_args

            init_args.setdefault("num_classes", int(num_classes_value))

        for v in node.values():
            inject_num_classes_into_eomt(v, num_classes_value)

    elif isinstance(node, list):
        for v in node:
            inject_num_classes_into_eomt(v, num_classes_value)


def instantiate_from_cfg(node: Any) -> Any:
    """
    Recursively instantiate objects from LightningCLI-style dicts:
      {"class_path": "...", "init_args": {...}}
    """
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


def inject_img_size_into_vit(node: Any, img_size_value: Any) -> None:
    if isinstance(node, dict):
        class_path = node.get("class_path", "")
        if class_path.split(".")[-1] == "ViT":
            init_args = node.get("init_args", None)
            if not isinstance(init_args, dict):
                init_args = {}
                node["init_args"] = init_args

            init_args.setdefault("img_size", img_size_value)

        for v in node.values():
            inject_img_size_into_vit(v, img_size_value)

    elif isinstance(node, list):
        for v in node:
            inject_img_size_into_vit(v, img_size_value)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default=[
            "D:/DAAI project/MaskArchitectureAnomaly_CourseProject-main/MaskArchitectureAnomaly_CourseProject-main/Validation_Dataset/Validation_Dataset/RoadAnomaly21/images/*.png"
        ],
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="MSP",
        choices=["MSP", "MaxL", "MaxE", "RbA", "MSP-T"],
        help="Anomaly scoring method (MSP, MaxL, MaxE, RbA, MSP-T)",
    )  # Method argument to decide the post-hoc method

    parser.add_argument(
        "--loadDir",
        default=r"C:\Users\karee\Desktop\Anomaly-Project\MaskArchitectureAnomaly_CourseProject",
    )

    parser.add_argument(
        "--eomt_config",
        default="eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml",
    )  # Relative path to config file

    parser.add_argument(
        "--eomt_ckpt", default="eomt_checkpoints/eomt_cityscapes.bin"
    )  # Relative path to pretrained model

    parser.add_argument(
        "--eomt_variant", default="base"
    )  # optional label for you (base/large)
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
        help="If set, saves the raw logits for each image",
    )

    parser.add_argument(
        "--logits_dir",
        default="logits",
    )  # Saved logits directory

    parser.add_argument(
        "--tempScale",
        type=float,
        default=1.0,
        help="Temperature scaling factor for softmax (only for MSP-T method)",
    )  # Temperature scale argument to decide post-processing in MSP-T

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []

    VIS_OUTPUT_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "vis_eomt", args.method
    )
    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)
    print(f"Visualization results will be saved to: {VIS_OUTPUT_DIR}")

    if not os.path.isabs(args.eomt_config):
        args.eomt_config = os.path.normpath(
            os.path.join(PROJECT_ROOT, args.eomt_config)
        )

    if not os.path.isabs(args.eomt_ckpt):
        args.eomt_ckpt = os.path.normpath(os.path.join(PROJECT_ROOT, args.eomt_ckpt))

    print("Resolved config:", args.eomt_config)
    print("Resolved ckpt:", args.eomt_ckpt)
    print("Config exists?", os.path.exists(args.eomt_config))
    print("Ckpt exists?", os.path.exists(args.eomt_ckpt))

    if not os.path.exists("results.txt"):
        open("results.txt", "w").close()
    file = open("results.txt", "a")

    device = torch.device("cpu" if args.cpu else "cuda")

    print("Loading EoMT config:", args.eomt_config)
    print("Loading EoMT ckpt:", args.eomt_ckpt)

    cfg = OmegaConf.load(args.eomt_config)

    # --- get img_size + num_classes ---
    data_args = cfg.get("data", {}).get("init_args", {})
    img_size = data_args.get("img_size", 640)
    num_classes = data_args.get("num_classes", 19)

    # If yaml doesn't provide img_size correctly, infer from filename
    if img_size is None:
        if "1024" in os.path.basename(args.eomt_config):
            img_size = 1024
        elif "640" in os.path.basename(args.eomt_config):
            img_size = 640
        else:
            img_size = 640  # safe default

    # Normalize img_size to tuple for ViT
    if isinstance(img_size, int):
        img_size_vit = (img_size, img_size)
    elif isinstance(img_size, (list, tuple)) and len(img_size) == 2:
        img_size_vit = tuple(img_size)
    else:
        img_size_vit = (640, 640)

    if num_classes is None:
        num_classes = 19  # Cityscapes semantic classes
    EVAL_SIZE = (512, 1024)  # keep your evaluation size (GT size)

    input_transform = Compose(
        [
            Resize(
                img_size_vit, Image.BILINEAR
            ),  # <-- model input MUST match (640,640)
            ToTensor(),
        ]
    )

    target_transform = Compose(
        [
            Resize(EVAL_SIZE, Image.NEAREST),  # keep GT at 512x1024 (as before)
        ]
    )

    # --- build net_node from YAML ---
    net_node = OmegaConf.to_container(
        cfg["model"]["init_args"]["network"], resolve=True
    )
    inject_img_size_into_vit(net_node, img_size_vit)
    inject_num_classes_into_eomt(net_node, num_classes)

    # Instantiate model
    model = instantiate_from_cfg(net_node)
    # 3) Load checkpoint
    ckpt = torch.load(args.eomt_ckpt, map_location="cpu", weights_only=True)
    print("Checkpoint type:", type(ckpt))

    state = None

    if isinstance(ckpt, dict):
        print("Checkpoint keys:", list(ckpt.keys())[:20])

        # try common keys where weights are stored
        for key in ["state_dict", "model", "model_state_dict", "net", "network"]:
            if key in ckpt:
                state = ckpt[key]
                print("Using weights from key:", key)
                break

        # if none of those keys exist, maybe ckpt itself is the state_dict
        if state is None:
            state = ckpt
            print("Using checkpoint dict directly as state_dict.")
    else:
        # ckpt is already a state_dict (rare but possible)
        state = ckpt
        print("Checkpoint is not a dict; using it directly as state_dict.")

    if not isinstance(state, dict):
        raise TypeError(f"Loaded weights are not a dict. Got: {type(state)}")

    # 4) Clean common prefixes (Lightning / DataParallel)
    def strip_prefix(k: str) -> str:
        for p in ("model.network.", "network.", "model.", "module."):
            if k.startswith(p):
                return k[len(p) :]
        return k

    clean_state = {strip_prefix(k): v for k, v in state.items()}

    # ---- FIX pos_embed mismatch by resizing checkpoint pos_embed to model size ----
    model_sd = model.state_dict()
    pe_key = "encoder.backbone.pos_embed"

    if pe_key in clean_state and pe_key in model_sd:
        if clean_state[pe_key].shape != model_sd[pe_key].shape:
            print(
                "pos_embed mismatch:",
                "ckpt",
                tuple(clean_state[pe_key].shape),
                "model",
                tuple(model_sd[pe_key].shape),
            )

            new_tokens = model_sd[pe_key].shape[1]
            clean_state[pe_key] = resize_pos_embed_patch_only(
                clean_state[pe_key], new_tokens
            )

            print("Resized pos_embed to", tuple(clean_state[pe_key].shape))
    # ------------------------------------------------------------------------------

    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    print("Loaded EoMT weights.")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))
    model.to(device)
    model.eval()

    # --- Debug counters ---
    total_imgs = 0
    missing_gt = 0
    skipped_no_ood = 0
    used_imgs = 0

    # optional: show only first N debug prints (avoid spam)
    debug_pairs_to_print = 5
    printed_pairs = 0
    LOGITS_OUTPUT_DIR = os.path.join(args.loadDir, args.logits_dir)
    if args.save_logits:
        if args.method == "MSP":
            os.makedirs(LOGITS_OUTPUT_DIR, exist_ok=True)
            print(f"Logits will be saved to: {LOGITS_OUTPUT_DIR}")
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        total_imgs += 1
        base_filename = os.path.basename(path).split(".")[0]
        logits_path = os.path.join(LOGITS_OUTPUT_DIR, f"{base_filename}.pt")
        print(path)
        images = (
            input_transform(Image.open(path).convert("RGB"))
            .unsqueeze(0)
            .float()
            .to(device)
        )

        with torch.no_grad():

            # If MSP-T then load the saved logits to save time
            if args.method == "MSP-T":
                base_filename = os.path.basename(path).split(".")[0]

                logit_file_path = os.path.join(
                    args.logits_dir, f"{base_filename.strip()}.pt"
                )

                # Load dictionary
                loaded_dict = torch.load(logit_file_path, map_location=device)
                mask_logits = loaded_dict["mask_logits"].to(device)
                class_logits = loaded_dict["class_logits"].to(device)

            # Else, evaluate the model to get results
            else:
                mask_logits_layers, class_logits_layers = model(images)
                mask_logits = mask_logits_layers[-1]
                class_logits = class_logits_layers[-1]

            # Save logits only in MSP case (not necessary but to avoid saving every single time on all methods)
            if args.save_logits:
                if args.method == "MSP":
                    torch.save(
                        {
                            "mask_logits": mask_logits.cpu(),
                            "class_logits": class_logits.cpu(),
                        },
                        logits_path,
                    )
                    print(f"Saved logits to: {logits_path}")

            # If MSP, normalize the class_logits by temperature scale
            if args.method == "MSP-T":
                t = (torch.ones(1) * args.tempScale).cuda()
                class_probs_all = F.softmax(class_logits / t, dim=-1)[..., :-1]
                mask_probs = mask_logits.sigmoid()

            # Else, continue
            else:
                class_probs_all = F.softmax(class_logits, dim=-1)[..., :-1]
                mask_probs = mask_logits.sigmoid()

            # Construct semantic map including the VOID channel
            sem_probs_all = torch.einsum(
                "bqk, bqhw -> bkhw", class_probs_all, mask_probs
            )

            # Normalize spatially across all class possibilities
            sem_probs_all = sem_probs_all / (
                sem_probs_all.sum(dim=1, keepdim=True) + 1e-6
            )

            # Resize to target resolution
            sem_probs_all = F.interpolate(
                sem_probs_all,
                size=(512, 1024),
                mode="bilinear",
                align_corners=False,
            )

            # In-Distribution probs
            sem_probs_id = sem_probs_all[:, :-1, :, :]

            if args.method == "MSP-T":
                anomaly_result = (
                    (1.0 - sem_probs_id.max(dim=1).values).squeeze().cpu().numpy()
                )
            elif args.method == "RbA":
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
                anomaly_result = rba_map.squeeze(0).cpu().numpy()  # [H, W]

            # Standard methods (MSP, MaxL, MaxE) work on in-distribution    channels
            elif args.method == "MSP":
                anomaly_result = (
                    (1.0 - sem_probs_id.max(dim=1).values).squeeze().cpu().numpy()
                )
            elif args.method == "MaxL":
                pixel_logits = torch.einsum(
                    "bqk, bqhw -> bkhw", class_logits[:, :, :-1], mask_probs
                )

                pixel_logits = F.interpolate(
                    pixel_logits,
                    size=(512, 1024),
                    mode="bilinear",
                    align_corners=False,
                )
                anomaly_result = (
                    (-pixel_logits.max(dim=1).values).squeeze().cpu().numpy()
                )
            elif args.method == "MaxE":
                # Shannon Entropy
                anomaly_result = (
                    (-(sem_probs_id * torch.log(sem_probs_id + 1e-12)).sum(dim=1))
                    .squeeze()
                    .cpu()
                    .numpy()
                )

        # --- Load GT ---
        pathGT = path.replace("images", "labels_masks")
        # Adapt for different image extensions
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

            # --- Debug: verify GT path exists ---
        if printed_pairs < debug_pairs_to_print:
            print(f"\nIMG: {path}")
            print(f"GT : {pathGT}")
            print("GT exists?", os.path.exists(pathGT))
            printed_pairs += 1

        if not os.path.exists(pathGT):
            missing_gt += 1
            # print the missing ones (optional)
            print("!! Missing GT file:", pathGT)
            continue

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)
        if printed_pairs <= debug_pairs_to_print:

            u = np.unique(ood_gts)
            print(
                "Raw GT unique values (first few):",
                u[:30],
                "..." if len(u) > 30 else "",
            )

        # RA: Unique labels: [0, 1, 2]  --> Remap 2 to 1
        # RA21, RO21, FS_LF, FS_Static: Unique labels: [0,1,255]  --> Remap 255 to 1
        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts == 2), 1, ood_gts)
        if "RoadObsticle21" in pathGT or "RoadAnomaly21" in pathGT:
            ood_gts = np.where((ood_gts == 255), 1, ood_gts)
        if "FS_LostFound_full" in pathGT or "fs_static" in pathGT:
            ood_gts = np.where((ood_gts == 255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            skipped_no_ood += 1
            continue
        else:
            used_imgs += 1
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
        del anomaly_result, ood_gts, mask
        torch.cuda.empty_cache()

    file.write("\n")
    print("\n--- Dataset filtering summary ---")
    print("Total images found:", total_imgs)
    print("Missing GT files   :", missing_gt)
    print("Skipped (no OOD=1)  :", skipped_no_ood)
    print("Used for metrics    :", used_imgs)

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
