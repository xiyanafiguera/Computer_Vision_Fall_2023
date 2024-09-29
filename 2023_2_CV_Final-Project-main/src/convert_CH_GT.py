import json
import os.path as osp
from PIL import Image


OUTPUT_PATH = "out/GT_labels/CrowdHuman"
IMAGE_PATH = "data/CrowdHuman/CrowdHuman_Images"
ODGT_PATH = "data/CrowdHuman/annotation_train.odgt"


def convert_GT():
    # Read the file
    with open(ODGT_PATH) as odgt_file:
        for line in odgt_file:
            # data contains total bounding boxes for one image file
            data = json.loads(line)

            # Load the image corresponding to the data, and get the width and height
            image_path = osp.join(IMAGE_PATH, data["ID"] + ".jpg")
            image = Image.open(image_path)
            image_width, image_height = image.size

            # Define a file for the labels
            GT_file_name = data["ID"] + ".txt"
            GT_file_path = osp.join(OUTPUT_PATH, GT_file_name)

            with open(GT_file_path, "w") as GT_file:
                # Convert the bounding boxes to YOLO format
                for bbox in data["gtboxes"]:
                    # bbox contains one bounding box
                    # bbox["fbox"] contains the bounding box coordinates
                    # bbox["fbox"][0] contains the x-coordinate of the top left corner
                    # bbox["fbox"][1] contains the y-coordinate of the top left corner
                    # bbox["fbox"][2] contains the width of the bounding box
                    # bbox["fbox"][3] contains the height of the bounding box

                    left_top_x = bbox["fbox"][0]
                    left_top_y = bbox["fbox"][1]
                    width = bbox["fbox"][2]
                    height = bbox["fbox"][3]

                    # YOLO format uses the center coordinates
                    center_x = left_top_x + width / 2
                    center_y = left_top_y + height / 2

                    # YOLO format uses the normalized coordinates (between 0 and 1)
                    center_x = center_x / image_width
                    center_y = center_y / image_height
                    width = width / image_width
                    height = height / image_height

                    # YOLO format: class center_x center_y width height score
                    GT_file.write(f"0 {center_x} {center_y} {width} {height} 1\n")


if __name__ == "__main__":
    convert_GT()
