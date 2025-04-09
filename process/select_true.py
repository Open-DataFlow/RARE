import json
import re
import argparse
import os
import shutil
from pathlib import Path

def extract_option(pred):
    # 1. get A/B/C/D
    for pattern in [
        r"<answer>(.*?)</answer>",
        r"<answer>(.*?)<answer>",
        r"^([A-Z])[.,:]",
        r"Answer:\s*([A-Z])\s*",
    ]:
        match = re.search(pattern, pred, re.DOTALL)
        if match is not None:
            pred = match.group(1)
    # 2. remove <>
    pred = pred.replace("<", "").replace(">", "")
    pred = pred.strip()
    return pred

def copy_images(image_paths, src_dir, dest_dir):

    os.makedirs(dest_dir, exist_ok=True)

    copied_images = {}

    for img_path in image_paths:
        if os.path.isabs(img_path):
            img_file = os.path.basename(img_path)
            src_img = img_path
        else:
            img_file = os.path.basename(img_path)
            src_img = os.path.join(src_dir, img_file)

        dest_img = os.path.join(dest_dir, img_file)

        if os.path.exists(src_img):
            shutil.copy2(src_img, dest_img)
            copied_images[img_path] = os.path.join(os.path.basename(dest_dir), img_file)
        else:
            print(f"Warning: Image not found: {src_img}")
            copied_images[img_path] = img_path

    return copied_images

def filter_correct_predictions(file_path, is_mm_mode=False):
    with open(file_path, "r") as f:
        data = json.load(f)

    correct_data = []
    image_paths = []

    for item in data:
        output = extract_option(item["output"])
        if is_mm_mode:
            if (
                "messages" in item
                and len(item["messages"]) > 1
                and "content" in item["messages"][1]
            ):
                predict = extract_option(item["messages"][1]["content"])
            else:
                continue
        else:
            predict = extract_option(item["predict"])

        if output == predict:
            correct_data.append(item)

            if "images" in item and isinstance(item["images"], list):
                image_paths.extend(item["images"])

    file_path_obj = Path(file_path)
    output_file = str(file_path_obj.with_stem(file_path_obj.stem + "_true"))

    if is_mm_mode and image_paths:

        parent_dir = "data"

        if "train" in file_path:
            src_img_dir = os.path.join(parent_dir, "train_document_images")
            dest_img_dir = os.path.join(parent_dir, "train_document_images_true")
        elif any(x in file_path for x in ["test", "val"]):
            src_img_dir = os.path.join(parent_dir, "val_document_images")
            dest_img_dir = os.path.join(parent_dir, "val_document_images_true")
        else:
            src_img_dir = os.path.join(parent_dir, "document_images")
            dest_img_dir = os.path.join(parent_dir, "document_images_true")

        path_mapping = copy_images(image_paths, src_img_dir, dest_img_dir)

        for item in correct_data:
            if "images" in item and isinstance(item["images"], list):
                item["images"] = [path_mapping.get(img_path, img_path) for img_path in item["images"]]

        print(f"Copied {len(path_mapping)} unique images to {dest_img_dir}")

    with open(output_file, "w") as f:
        json.dump(correct_data, f, indent=2)

    print(f"Filtered {len(correct_data)} correct predictions out of {len(data)} total")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter correct predictions from a dataset"
    )
    parser.add_argument("file_path", help="Path to the JSON dataset file")
    parser.add_argument(
        "--mm",
        action="store_true",
        help="Use messages[1].content instead of predict and handle image paths"
    )
    args = parser.parse_args()
    filter_correct_predictions(args.file_path, args.mm)
