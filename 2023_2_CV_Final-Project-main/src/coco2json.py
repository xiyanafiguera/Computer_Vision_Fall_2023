import os
import glob
import os.path as osp
import json
import argparse
from PIL import Image

IMAGE_PATH = "YOLOv6/runs/inference/exp/images/"
LABEL_PATH = "YOLOv6/runs/inference/exp/images/labels/"
OUTPUT_PATH = "out/testing_results/"


def main(args):
    # Make output path directory
    os.makedirs(args.output_path, exist_ok=True)
    json_file = osp.join(args.output_path, "pred.json")

    # Get image and label file list
    image_files = sorted(glob.glob(osp.join(args.image_path, "*.jpg")))
    label_files = sorted(glob.glob(osp.join(args.label_path, "*.txt")))

    json_list = []

    for image_file, label_file in zip(image_files, label_files):
        # Get data index
        data_index = label_file.split("/")[-1].split(".")[0]

        # Read image file and get width & height of image
        img = Image.open(image_file)
        image_width, image_height = img.size

        # print(f"data_index: {data_index}, width: {width}, height: {height}")

        # Read label file and convert into json format
        with open(label_file, "r") as f:
            lines = f.readlines()
            box_infos = [line.split(" ") for line in lines]

            # Convert box_infos into json format
            for box_info in box_infos:
                box_info = [float(x) for x in box_info]

                # box_info: [class, center_x, center_y, width, height, score]
                # x, y, w, h are normalized value (0 ~ 1)
                x, y, w, h, s = box_info[1:]

                # convert x,y,w,h to x1,y1,x2,y2
                x1 = (x - w / 2) * image_width
                y1 = (y - h / 2) * image_height
                x2 = (x + w / 2) * image_width
                y2 = (y + h / 2) * image_height
                # cls_id starts from 0
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)

                # Make a dictionary object for json format
                json_dict = {
                    "image_id": int(data_index),
                    "category_id": 1,
                    "bbox": [x1, y1, w, h],
                    "score": s,
                }

                # Add json_dict into json_list
                json_list.append(json_dict)

    # Save json_list into json file
    with open(json_file, "w") as f:
        json.dump(json_list, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", "-ip", type=str, default=IMAGE_PATH)
    parser.add_argument("--label_path", "-lp", type=str, default=LABEL_PATH)
    parser.add_argument("--output_path", "-op", type=str, default=OUTPUT_PATH)

    args = parser.parse_args()

    main(args)
