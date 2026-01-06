# Code with dataset loader for VOC12 and Cityscapes (adapted from bodokaiser/piwise code)
# Sept 2017
# Eduardo Romera
#######################

import numpy as np
import os
import os.path as osp

from PIL import Image

from torch.utils.data import Dataset

EXTENSIONS = [".jpg", ".png", "webp"]


def load_image(file):
    return Image.open(file)


def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)


def is_label(filename):
    return filename.endswith("_labelTrainIds.png")


def image_path(root, basename, extension):
    return os.path.join(root, f"{basename}{extension}")


def image_path_city(root, name):
    return os.path.join(root, f"{name}")


def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])


class VOC12(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, "images")
        self.labels_root = os.path.join(root, "labels")

        self.filenames = [
            image_basename(f) for f in os.listdir(self.labels_root) if is_image(f)
        ]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, ".jpg"), "rb") as f:
            image = load_image(f).convert("RGB")
        with open(image_path(self.labels_root, filename, ".png"), "rb") as f:
            label = load_image(f).convert("P")

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)


class cityscapes(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None, subset="val"):

        self.images_root = os.path.join(root, "leftImg8bit/" + subset)
        self.labels_root = os.path.join(root, "gtFine/" + subset)
        print(self.images_root, self.labels_root)
        self.filenames = [
            os.path.join(dp, f)
            for dp, dn, fn in os.walk(os.path.expanduser(self.images_root))
            for f in fn
            if is_image(f)
        ]
        self.filenames.sort()

        self.filenamesGt = [
            os.path.join(dp, f)
            for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root))
            for f in fn
            if is_label(f)
        ]
        self.filenamesGt.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        # print(filename)

        with open(image_path_city(self.images_root, filename), "rb") as f:
            image = load_image(f).convert("RGB")
        with open(image_path_city(self.labels_root, filenameGt), "rb") as f:
            label = load_image(f).convert("P")

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label, filename, filenameGt

    def __len__(self):
        return len(self.filenames)


class Anomaly(Dataset):
    def __init__(self, root, input_transform=None, target_transform=None):

        self.images_root = os.path.join(root, "images")
        self.labels_root = os.path.join(root, "labels_masks")
        print(self.images_root, self.labels_root)
        self.filenames = [
            osp.join(self.images_root, f)
            for f in os.listdir(self.images_root)
            if is_image(f)
        ]
        self.filenames.sort()

        self.filenamesGt = [
            osp.join(self.labels_root, f.replace("images", "labels_masks"))
            for f in os.listdir(self.images_root)
            if is_image(f)
        ]
        self.filenamesGt.sort()

        filenames_base = [image_basename(f) for f in self.filenames]
        self.filenamesGt = [
            osp.join(self.labels_root, f + ".png") for f in filenames_base
        ]

        print(
            f"Loaded {len(self.filenames)} images and {len(self.filenamesGt)} masks from {root}"
        )

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        # Handle mask extension mismatch (common in anomaly datasets)
        if not osp.exists(filenameGt):
            # Attempt to use .webp extension for mask if .png is not found
            filenameGt_webp = filenameGt.replace(".png", ".webp")
            if osp.exists(filenameGt_webp):
                filenameGt = filenameGt_webp
            # Handle jpg to png conversion for RoadAnomaly (used in your evalAnomaly.py)
            elif filename.endswith(".jpg"):
                filenameGt = filenameGt.replace(".jpg", ".png")

        with open(filename, "rb") as f:
            image = load_image(f).convert("RGB")
        with open(filenameGt, "rb") as f:
            label = load_image(f).convert("P")

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label, filename, filenameGt

    def __len__(self):
        return len(self.filenames)
