import os
import glob
import numpy as np
from PIL import Image
from argparse import ArgumentParser


def inspect_labels(directory_path):
    # Support common image formats
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(directory_path, ext)))

    if not image_files:
        print(f"No images found in {directory_path}")
        return

    print(f"Found {len(image_files)} images. Analyzing unique pixel values...")

    global_unique_values = set()

    for path in image_files:
        try:
            # Open image and convert to numpy array without any resizing/transformations
            # to see the raw, original labels
            img = Image.open(path)
            img_np = np.array(img)

            unique_in_img = np.unique(img_np)
            global_unique_values.update(unique_in_img)

            # Optional: Print per-image values if the dataset is small
            # print(f"{os.path.basename(path)}: {unique_in_img}")

        except Exception as e:
            print(f"Could not process {path}: {e}")

    print("-" * 30)
    print(
        f"Unique Label Values found in directory: \n{sorted(list(global_unique_values))}"
    )
    print("-" * 30)

    # Contextual help for SMIYC datasets
    print("\nCommon Mapping Guide for RoadObstacle21/RoadAnomaly21:")
    print(" - 0: Usually In-Distribution (Road/Background)")
    print(" - 1 or 2: Usually Anomaly/Obstacle")
    print(" - 255: Usually Void/Ignore (Not evaluated)")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dir", help="Path to the directory containing label masks")
    args = parser.parse_args()

    inspect_labels(args.dir)
