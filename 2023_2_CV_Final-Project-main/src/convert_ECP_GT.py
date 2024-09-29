import os
import json

OUTPUT_PATH = "out/GT_labels/ECP/train"
ORIGINAL_PATH = "data/EuroCityPersons/GT_labels/train"


def convert_ECP_GT():
    # Read the file
    for city_dir in os.listdir(ORIGINAL_PATH):
        if city_dir == ".DS_Store":
            continue

        os.makedirs(os.path.join(OUTPUT_PATH, city_dir), exist_ok=True)
        for filename in os.listdir(os.path.join(ORIGINAL_PATH, city_dir)):
            with open(os.path.join(ORIGINAL_PATH, city_dir, filename)) as json_file:
                # data contains total bounding boxes for one image file
                data = json.load(json_file)

                image_width = data["imagewidth"]
                image_height = data["imageheight"]
                bboxs = data["children"]

                GT_file_name = filename.split(".")[0] + ".txt"
                GT_file_path = os.path.join(OUTPUT_PATH, city_dir, GT_file_name)
                with open(GT_file_path, "w") as GT_file:
                    # Convert the bounding boxes to YOLO format
                    for bbox in bboxs:
                        # bbox contains one bounding box

                        # Only use the bounding boxes with identity "pedestrian"
                        if bbox["identity"] == "pedestrian":
                            left_top_x = bbox["x0"]
                            left_top_y = bbox["y0"]
                            right_bottom_x = bbox["x1"]
                            right_bottom_y = bbox["y1"]

                            width = right_bottom_x - left_top_x
                            height = right_bottom_y - left_top_y
                            center_x = left_top_x + width / 2
                            center_y = left_top_y + height / 2

                            # YOLO format uses the normalized coordinates (between 0 and 1)
                            center_x = center_x / image_width
                            center_y = center_y / image_height
                            width = width / image_width
                            height = height / image_height

                            # YOLO format: class center_x center_y width height score
                            GT_file.write(
                                f"0 {center_x} {center_y} {width} {height} 1\n"
                            )


if __name__ == "__main__":
    convert_ECP_GT()
