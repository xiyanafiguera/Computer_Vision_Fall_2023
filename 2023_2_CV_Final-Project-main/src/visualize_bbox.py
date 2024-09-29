"""
Visualize bounding box of images

Usage:
    python src/visualize_bbox.py -d <dataset_type> -s <score>
    # dataset_type: CH (CrowdHuman), ECP (EuroCityPersons), IN (Indoor)
    # score: score threshold (default: 0.5)

Example:
    python src/visualize_bbox.py -d CH -s 0.5
"""

import glob
import os.path as osp
import os
from PIL import Image, ImageDraw
import argparse

from utils.types import DatasetType, VisualizeBBoxArgs
from utils.data_config import DataConfig


def main(args: VisualizeBBoxArgs):
    # Set the dataset configuration (CrowdHuman, EuroCityPersons, Indoor)
    data_config = DataConfig(args.dataset_type)

    # Get image and label file list
    # I only use 100 images for test (You can freely change this number)
    image_files = sorted(glob.glob(osp.join(data_config.IMAGE_PATH, "*.jpg")))
    label_files = sorted(glob.glob(osp.join(data_config.LABEL_PATH, "*.txt")))

    # Make output path directory
    os.makedirs(data_config.VIZ_BBOX_OUTPUT_PATH, exist_ok=True)

    for image_file, label_file in zip(image_files, label_files):
        # Read image file and define PIL draw object
        img = Image.open(image_file)
        draw = ImageDraw.Draw(img)

        # Read width & height of image, and define line width of bounding box
        width, height = img.size
        line_width = 1
        if width > 1000:
            line_width = 2
        elif width > 2000:
            line_width = 4
        elif width > 4000:
            line_width = 8

        # Read label file and draw bounding box
        with open(label_file, "r") as f:
            lines = f.readlines()
            box_infos = [line.split(" ") for line in lines]

            # Draw bounding box
            for box_info in box_infos:
                box_info = [float(x) for x in box_info]

                # box_info: [class, center_x, center_y, width, height, score]
                # x, y, w, h are normalized value (0 ~ 1)
                x, y, w, h, s = box_info[1:]
                x = int(x * width)
                y = int(y * height)
                w = int(w * width)
                h = int(h * height)

                # Draw bounding box if score is higher than threshold (default: 0.5)
                if s > args.score:
                    draw.rectangle(
                        ((x - w / 2, y - h / 2), (x + w / 2, y + h / 2)),
                        outline=(255, 0, 0),
                        width=line_width,
                    )

        # Save image file
        img.save(osp.join(data_config.VIZ_BBOX_OUTPUT_PATH, osp.basename(image_file)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", "-s", type=float, default=0.5)
    parser.add_argument(
        "--dataset_type",
        "-d",
        type=DatasetType,
        help=f"Please enter the dataset type ({[d.value for d in DatasetType]})",
    )
    args: VisualizeBBoxArgs = parser.parse_args()

    main(args)
