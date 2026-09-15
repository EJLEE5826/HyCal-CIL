import argparse
import os

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dataset_dir = os.path.join(args.data_root, "Galaxy10_DECals")
    h5_path = os.path.join(dataset_dir, "Galaxy10_DECals.h5")
    output_dir = os.path.join(dataset_dir, "_galaxy_temp")
    output_img_dir = os.path.join(output_dir, "images")
    output_csv_path = os.path.join(output_dir, "galaxy10_split.csv")
    image_key = "images"
    label_key = "ans"

    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"Input file does not exist: {h5_path}")
    if os.path.lexists(output_dir):
        raise FileExistsError(f"Output already exists: {output_dir}")

    os.makedirs(output_img_dir)

    with h5py.File(h5_path, "r") as f:
        num_samples = f[image_key].shape[0]
        labels = np.array(f[label_key])

    indices = np.arange(num_samples)
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=42
    )

    split_map = {}
    for i in train_idx:
        split_map[i] = "train"
    for i in val_idx:
        split_map[i] = "val"
    for i in test_idx:
        split_map[i] = "test"

    csv_rows = []
    with h5py.File(h5_path, "r") as f:
        img_dataset = f[image_key]

        for idx in tqdm(range(num_samples), desc="Saving images"):
            img_array = img_dataset[idx]
            img = Image.fromarray(img_array)
            filename = f"{idx:05d}.jpg"
            img.save(os.path.join(output_img_dir, filename))

            split = split_map[idx]
            label = int(labels[idx])
            csv_rows.append([split, filename, label])
            print(split, filename, label)

    df = pd.DataFrame(csv_rows, columns=["split", "filename", "class_num"])
    df.to_csv(output_csv_path, index=False)

    print(f"Images saved: {output_img_dir}")
    print(f"CSV saved: {output_csv_path}")


if __name__ == "__main__":
    main()
