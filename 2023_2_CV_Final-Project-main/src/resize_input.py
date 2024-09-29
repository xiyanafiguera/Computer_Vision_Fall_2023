"""
Resize the input image data (for student model) to 448x448
The ratio of width and height is not kept with the original image.

Since the YOLO label file contains normalized value (0 ~ 1),
We don't need to change the label file.

Usage:
    python resize_input.py -d <dataset_type>
    # dataset_type: CH (CrowdHuman), ECP (EuroCityPersons), IN (Indoor)

Example:
    python resize_input.py -d CH
"""

import glob
import os.path as osp
import os
import argparse
from PIL import Image

from utils.types import DatasetType, ResizeInputArgs
from utils.data_config import DataConfig


def main(args: ResizeInputArgs):
    # Set the dataset configuration (CrowdHuman, EuroCityPersons, Indoor)
    data_config = DataConfig(args.dataset_type)

    # Get image file list
    image_files = sorted(glob.glob(osp.join(data_config.IMAGE_PATH, "*.jpg")))

    # Make output path directory
    os.makedirs(data_config.RESIZED_OUTPUT_PATH, exist_ok=True)

    for image_file in image_files:
        # Get image file name & set the output path
        image_file_name = osp.basename(image_file)  # ex) 000000,xxxxxxxxxxxx.jpg
        output_path = osp.join(data_config.RESIZED_OUTPUT_PATH, image_file_name)

        # Read image file
        img = Image.open(image_file)

        # Resize image
        img = img.resize((448, 448))

        # Save image
        img.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_type",
        "-d",
        type=DatasetType,
        help=f"Please enter the dataset type ({[d.value for d in DatasetType]})",
    )
    args: ResizeInputArgs = parser.parse_args()

    main(args)
