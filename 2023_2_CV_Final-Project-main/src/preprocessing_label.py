"""
Please change the INPUT_PATH and OUTPUT_PATH to your own path.

Usage:
    python preprocessing_label.py -d IN+CH -l GT -t train
"""

import os
import argparse


def main(args):
    INPUT_PATH = (
        f"data/combined_dataset/{args.dataset}/{args.label_type}/{args.data_type}"
    )
    OUTPUT_PATH = (
        f"data/modified_dataset/{args.dataset}/{args.label_type}/{args.data_type}"
    )

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    files = sorted(os.listdir(INPUT_PATH))

    for file in files:
        file_path = os.path.join(INPUT_PATH, file)
        # print(file_path)
        with open(file_path, "r") as f:
            lines = f.readlines()
            valid_boxes = []
            for line in lines:
                box_info = [float(x) for x in line.split(" ")]
                if len(box_info) == 5 or box_info[-1] > 0.5:
                    box_info = box_info[:5]
                    x, y, w, h = box_info[1:]

                    min_x = x - w / 2
                    max_x = x + w / 2
                    min_y = y - h / 2
                    max_y = y + h / 2

                    if min_x < 0:
                        w -= min_x
                        x = x + min_x / 2
                    if max_x > 1:
                        w -= max_x - 1
                        x = x - (max_x - 1) / 2
                    if min_y < 0:
                        h -= min_y
                        y = y + min_y / 2
                    if max_y > 1:
                        h -= max_y - 1
                        y = y - (max_y - 1) / 2

                    x = max(0, x)
                    y = max(0, y)
                    w = max(0, w)
                    h = max(0, h)

                    box_info[1:] = [x, y, w, h]

                    valid_line = " ".join([str(x) for x in box_info]) + "\n"
                    valid_boxes.append(valid_line)

        with open(os.path.join(OUTPUT_PATH, file), "w") as f:
            f.writelines(valid_boxes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="IN+CH",
        choices=["IN+CH", "IN+ECP", "ECP+CH"],
    )
    parser.add_argument(
        "--label_type", "-l", type=str, default="GT", choices=["GT", "labels"]
    )
    parser.add_argument(
        "--data_type", "-t", type=str, default="train", choices=["train", "val"]
    )
    args = parser.parse_args()

    main(args)
